#!/usr/bin/env python3
"""Train the target -> procedure LM with MLX.

    python -m lm.train --preset small --epochs 8 --out models/small

Loss is computed on procedure tokens only (after <sep>), not on the prompt.
"""

import argparse
import json
import math
import os
import shutil
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from lm.model import PRESETS, GPT, Config, save

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAD_TO = 32  # pad lengths to a multiple of this to bound compiled shapes


class Split:
    def __init__(self, path):
        z = np.load(path)
        self.tokens, self.offsets, self.plens = z["tokens"], z["offsets"], z["plens"]
        self.lens = np.diff(self.offsets)

    def __len__(self):
        return len(self.plens)

    def batch(self, idx):
        L = int(self.lens[idx].max())
        L = (L + PAD_TO - 1) // PAD_TO * PAD_TO
        x = np.zeros((len(idx), L), dtype=np.int32)
        m = np.zeros((len(idx), L - 1), dtype=np.float32)
        for r, i in enumerate(idx):
            s, e = self.offsets[i], self.offsets[i + 1]
            x[r, : e - s] = self.tokens[s:e]
            m[r, self.plens[i] - 1 : e - s - 1] = 1.0  # predict tokens after <sep>
        return mx.array(x), mx.array(m)

    def epoch_batches(self, bs, rng):
        """Length-bucketed shuffled batches (sort within chunks of 64 batches)."""
        perm = rng.permutation(len(self))
        chunk = bs * 64
        batches = []
        for c in range(0, len(perm), chunk):
            part = perm[c : c + chunk]
            part = part[np.argsort(self.lens[part], kind="stable")]
            batches += [part[b : b + bs] for b in range(0, len(part), bs)]
        rng.shuffle(batches)
        return batches


def loss_fn(model, x, m):
    logits, _ = model(x[:, :-1])
    ce = nn.losses.cross_entropy(logits, x[:, 1:], reduction="none")
    return (ce * m).sum() / m.sum()


def evaluate(model, split, bs=512, max_examples=20000):
    idx = np.arange(min(len(split), max_examples))
    tot, n = 0.0, 0.0
    for b in range(0, len(idx), bs):
        x, m = split.batch(idx[b : b + bs])
        logits, _ = model(x[:, :-1])
        ce = nn.losses.cross_entropy(logits, x[:, 1:], reduction="none")
        tot += (ce * m).sum().item()
        n += m.sum().item()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data/lm"))
    ap.add_argument("--preset", default="small", choices=PRESETS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=8)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--min-lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-gb", type=float, default=32,
                    help="MLX memory cap (GB); keeps the GPU from starving macOS")
    ap.add_argument("--pause-ms", type=float, default=0,
                    help="sleep between steps to leave GPU time to the system")
    args = ap.parse_args()

    # Safety (2026-11-10: three concurrent MLX runs hung the GPU and the
    # kernel watchdog rebooted the Mac): one training at a time, capped memory.
    lock_path = os.path.join(ROOT, "models", ".train.lock")
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    lock = open(lock_path, "w")
    try:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another lm.train is running (models/.train.lock): refusing to start")
    mx.set_memory_limit(int(args.mem_gb * 2**30))
    mx.set_cache_limit(int(args.mem_gb * 2**30 / 4))

    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    with open(os.path.join(args.data, "meta.json")) as fh:
        meta = json.load(fh)
    train, val = Split(os.path.join(args.data, "train.npz")), Split(os.path.join(args.data, "val.npz"))
    cfg = Config(vocab=meta["vocab"], max_len=meta["max_len"], **PRESETS[args.preset])
    model = GPT(cfg)
    mx.eval(model.parameters())
    n_params = model.n_params()

    steps_per_epoch = math.ceil(len(train) / args.bs)
    total = int(steps_per_epoch * args.epochs)
    sched = optim.join_schedules(
        [optim.linear_schedule(1e-7, args.lr, args.warmup),
         optim.cosine_decay(args.lr, total - args.warmup, args.min_lr)],
        [args.warmup])
    opt = optim.AdamW(learning_rate=sched, weight_decay=args.wd, betas=[0.9, 0.95])

    state = [model.state, opt.state]

    def _step(x, m):
        loss, grads = nn.value_and_grad(model, loss_fn)(model, x, m)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        return loss

    step_c = mx.compile(_step, inputs=state, outputs=state)

    os.makedirs(args.out, exist_ok=True)
    log = open(os.path.join(args.out, "train_log.jsonl"), "a")
    print(f"preset={args.preset} params={n_params/1e6:.2f}M train={len(train)} "
          f"steps={total} ({steps_per_epoch}/epoch)", flush=True)

    best, it, t0, tok_seen, run_loss = float("inf"), 0, time.time(), 0, None
    while it < total:
        for idx in train.epoch_batches(args.bs, rng):
            if it >= total:
                break
            x, m = train.batch(idx)
            loss = step_c(x, m)
            mx.eval(state)
            if args.pause_ms:
                time.sleep(args.pause_ms / 1000)
            tok_seen += int(train.lens[idx].sum())
            lv = loss.item()
            run_loss = lv if run_loss is None else 0.98 * run_loss + 0.02 * lv
            it += 1
            if it % 100 == 0:
                el = time.time() - t0
                print(f"it {it}/{total} loss {run_loss:.4f} lr {float(sched(it)):.2e} "
                      f"{tok_seen/el/1e3:.0f}k tok/s {el/60:.1f} min", flush=True)
            if it % args.eval_every == 0 or it == total:
                model.eval()
                vl = evaluate(model, val)
                model.train()
                rec = {"it": it, "epoch": it / steps_per_epoch, "train_loss": run_loss,
                       "val_loss": vl, "minutes": (time.time() - t0) / 60}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print("EVAL", json.dumps(rec), flush=True)
                extra = {"preset": args.preset, "params": n_params, **rec}
                save(model, os.path.join(args.out, "last"), extra)
                shutil.copy(os.path.join(args.data, "vocab.json"), os.path.join(args.out, "last"))
                if vl < best:
                    best = vl
                    save(model, os.path.join(args.out, "best"), extra)
                    shutil.copy(os.path.join(args.data, "vocab.json"), os.path.join(args.out, "best"))
    print(f"DONE best val_loss {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
