// Drives the page. All chemistry logic lives in Python (the repository's own
// modules, under Pyodide) or in RDKit MinimalLib; this file only moves data.

const $ = (id) => document.getElementById(id);
const SOLVENTS_NOTE = 'solvent or ubiquitous reagent';

let pyodide = null, RDKit = null;
const loaded = new Set();

function boot(msg) { $('boot').textContent = msg; }

async function fetchBytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}
async function fetchText(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return await r.text();
}

async function start() {
  const t0 = performance.now();
  pyodide = await loadPyodide({
    indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.28.3/full/',
  });
  boot('Loading NumPy…');
  await pyodide.loadPackage('numpy');

  boot('Unpacking the published modules…');
  // The python/ tree is the repository's own code, copied verbatim by
  // web/build.py. Mount it by writing the files into Pyodide's filesystem.
  const manifest = JSON.parse(await fetchText('py/manifest.json'));
  for (const path of manifest) {
    const data = await fetchText('py/' + path);
    const dir = ('/py/' + path).split('/').slice(0, -1).join('/');
    pyodide.FS.mkdirTree(dir);
    pyodide.FS.writeFile('/py/' + path, data);
  }
  pyodide.runPython(`
import sys
for p in ("/py", "/py/lambdaproc"):
    if p not in sys.path:
        sys.path.insert(0, p)
import webgen
`);

  boot('Starting RDKit…');
  if (typeof initRDKitModule !== 'function') {
    throw new Error('RDKit MinimalLib did not load (initRDKitModule missing)');
  }
  RDKit = await initRDKitModule();

  boot(`Ready in ${((performance.now() - t0) / 1000).toFixed(1)} s. ` +
       'The model itself loads on first use.');
  $('go').disabled = false;
  $('go').textContent = 'Generate';
  depict();
}

async function ensureModel(name) {
  if (loaded.has(name)) return;
  $('go').textContent = `Fetching ${name}…`;
  const [cfg, vocab, weights] = await Promise.all([
    fetchText(`models/${name}/config.json`),
    fetchText(`models/${name}/vocab.json`),
    fetchBytes(`models/${name}/weights.safetensors`),
  ]);
  const load = pyodide.globals.get('webgen').load;
  load(name, cfg, vocab, weights);
  load.destroy();
  loaded.add(name);
}

// ------------------------------------------------------------------ RDKit
function canonTarget(smi) {
  const m = RDKit.get_mol(smi);
  if (!m) return null;
  const out = m.get_smiles();
  m.delete();
  return out;
}

function svgFor(smi, w = 140, h = 100) {
  const m = RDKit.get_mol(smi);
  if (!m) return '';
  let svg = '';
  try {
    // Transparent background needs the details form; fall back to the plain
    // one if this MinimalLib build does not accept it.
    svg = m.get_svg_with_highlights(JSON.stringify({
      width: w, height: h, bondLineWidth: 1, backgroundColour: [0, 0, 0, 0],
    }));
  } catch (e) {
    try { svg = m.get_svg(w, h); } catch (e2) { svg = ''; }
  }
  m.delete();
  return svg;
}

function depict() {
  if (!RDKit) return;
  const c = canonTarget($('target').value.trim());
  $('depict').innerHTML = c ? svgFor(c, 260, 170)
    : '<span style="color:var(--bad);font-size:13px">RDKit cannot parse this SMILES</span>';
}

// --------------------------------------------------------------- generate
async function generate() {
  const name = $('model').value;
  const target = $('target').value.trim();
  const out = $('out');

  const canon = canonTarget(target);
  if (!canon) {
    out.innerHTML = `<div class="note bad"><b>Unparseable SMILES.</b>
      RDKit could not read <code>${escapeHtml(target)}</code>.</div>`;
    return;
  }

  let shelf = null;
  const warnings = [];
  if (name === 'cond') {
    shelf = [];
    for (const line of $('shelf').value.split('\n')) {
      const s = line.trim();
      if (!s) continue;
      const c = canonTarget(s);
      if (c) shelf.push(c);
      else warnings.push(`Ignored unparseable shelf SMILES <code>${escapeHtml(s)}</code>.`);
    }
    if (shelf.length < 4) {
      warnings.push(`Only ${shelf.length} material(s) on the shelf. This model
        was trained on shelves of four or more and tends to drop reactants
        below that — add a few distractors.`);
    }
    if (!shelf.length) shelf = null;
  }

  $('go').disabled = true;
  try {
    await ensureModel(name);
    $('go').textContent = 'Generating…';
    out.innerHTML = '<div class="note">Sampling…</div>';
    await new Promise((r) => setTimeout(r, 20));

    const t0 = performance.now();
    const propose = pyodide.globals.get('webgen').propose;
    const raw = propose(name, canon, parseInt($('n').value) || 1,
                        parseFloat($('temp').value) || 0.0,
                        parseInt($('topk').value) || 0,
                        parseInt($('seed').value) || 0,
                        shelf ? pyodide.toPy(shelf) : undefined);
    propose.destroy();
    const cands = JSON.parse(raw);
    const secs = ((performance.now() - t0) / 1000).toFixed(2);

    // Drawable structure for each valid candidate, built by the same parser
    // that validated it.
    const gfn = pyodide.globals.get('webgen').graph;
    for (const c of cands) {
      if (c.valid) c.graph = JSON.parse(gfn(c.canonical));
    }
    gfn.destroy();

    // Cross-check: what does the target-only model propose, shelf unseen?
    let cross = null;
    if (name === 'cond' && shelf) {
      $('go').textContent = 'Cross-checking…';
      await ensureModel('micro18');
      const p2 = pyodide.globals.get('webgen').propose;
      const free = JSON.parse(p2('micro18', canon, 1, 0.0, 0, 0));
      p2.destroy();
      cross = free.find((c) => c.valid) || null;
    }
    render(cands, shelf, warnings, secs, cross);
  } catch (e) {
    out.innerHTML = `<div class="note bad"><b>Failed.</b>
      <code>${escapeHtml(String(e))}</code></div>`;
  } finally {
    $('go').disabled = false;
    $('go').textContent = 'Generate';
  }
}

// ----------------------------------------------------------------- graph
// The structure comes from procgraph.py, which reads the AST that
// lambdaproc itself parsed and keys nodes by the canonicaliser's own key.
// This function only draws it: it never looks at the document text.
function flowHtml(g, shelfSet) {
  if (!g || g.error) {
    return `<div class="note warn">Could not draw this one
      (${escapeHtml(g ? g.error : 'no graph')}). The text below is unaffected.</div>`;
  }
  const seen = new Map();   // node id -> the step number it was drawn as
  let step = 0;

  function feed(n) {
    if (n.kind === 'ref') {
      return `<div class="backref">↑ the stream from step
        ${seen.get(n.id) || '?'}</div>`;
    }
    if (n.kind === 'sym') {
      return `<div class="feed"><code class="f-txt">${escapeHtml(n.sym)}${
        n.index == null ? '' : n.index}</code></div>`;
    }
    const svg = svgFor(n.smiles, 92, 56);
    return `<div class="feed${shelfSet.has(n.smiles) ? ' on-shelf' : ''}">
      ${svg || ''}<div class="f-txt">${escapeHtml(n.smiles)}${
      n.index == null ? '' : `<div class="qty">${n.index}</div>`}</div></div>`;
  }

  // Post-order: a node is numbered after the streams feeding it, so the
  // numbers read in the order the procedure is carried out.
  function draw(n) {
    if (n.kind === 'ref') return `<div class="node">${feed(n)}</div>`;
    if (n.kind !== 'op') return `<div class="node">${feed(n)}</div>`;
    const streams = n.args.filter((a) => a.kind === 'op' || a.kind === 'ref');
    const inputs = n.args.filter((a) => a.kind !== 'op' && a.kind !== 'ref');
    const above = streams.map(draw).join('');
    seen.set(n.id, ++step);
    const connector = streams.length === 0 ? ''
      : (streams.length === 1
        ? '<div class="pipe arrow"></div>'
        : `<div class="join" style="width:${(streams.length - 1) * 60 + 2}px">
           </div><div class="pipe arrow"></div>`);
    return `<div class="node">
      ${streams.length ? `<div class="branches">${above}</div>` : ''}
      ${connector}
      <div class="opbox${n.shared ? ' shared' : ''}">
        <div class="ophead"><span class="opsym">${escapeHtml(n.sym)}</span>
          <span class="oplabel">${escapeHtml(n.label || '')}</span>
          ${n.index == null ? ''
            : `<span class="opparam">${n.index}</span>`}</div>
        ${inputs.length ? `<div class="feeds">${inputs.map(feed).join('')}</div>`
          : ''}
      </div></div>`;
  }

  const body = draw(g.root);
  const target = $('target').value.trim();
  const canon = canonTarget(target) || target;
  const shared = g.n_mentions > g.n_nodes;
  return `<div class="flow"><div class="flow-in">${body}
    <div class="pipe arrow"></div>
    <div class="product">${svgFor(canon, 150, 92)}
      <div class="f-txt">${escapeHtml(canon)}</div></div></div>
    <p class="flow-note">${step} operation${step > 1 ? 's' : ''}, read top to
      bottom. Numbers on a box are the operation's parameter (minutes, or the
      step index the corpus recorded); numbers under a structure are the
      quantity. ${shared ? 'Dashed boxes are reused streams: the document '
        + 'denotes a DAG, not a tree.' : ''}</p></div>`;
}

function materialsOf(doc) {
  const out = [];
  const re = /\(([a-z][\w-]*)\s+"([^"]*)"\)/g;
  let m;
  while ((m = re.exec(doc)) !== null) out.push(m[2]);
  return out;
}

function render(cands, shelf, warnings, secs, cross) {
  const valid = cands.filter((c) => c.valid);
  const shelfSet = new Set(shelf || []);
  let h = '';

  for (const w of warnings) h += `<div class="note warn">${w}</div>`;

  h += `<div class="stats">
    <div class="stat"><b>${cands.length}</b><span>distinct proposals</span></div>
    <div class="stat"><b>${valid.length}</b><span>passed the validator</span></div>
    <div class="stat"><b>${secs} s</b><span>in this tab</span></div>
  </div>`;

  if (cross && valid.length) {
    const mine = new Set(materialsOf(valid[0].canonical));
    const theirs = materialsOf(cross.canonical);
    const shared = theirs.filter((s) => mine.has(s));
    if (shared.length) {
      h += `<div class="note ok"><b>Cross-check.</b> The target-only model,
        which never saw your shelf, also proposes ${shared.length} of these
        materials. On test data, agreement goes with higher accuracy (0.84
        against 0.68 building-block recall).</div>`;
    } else {
      const off = theirs.filter((s) => !shelfSet.has(s)).slice(0, 4);
      h += `<div class="note warn"><b>Cross-check.</b> The target-only model
        proposes different materials. This happens on 62 % of adequate
        shelves too, so it is not proof of a problem — but if your shelf might
        lack the key building block, note that this model does not detect
        that: it substitutes the nearest thing you gave it, in a procedure
        that still validates.
        ${off.length ? 'Without your shelf it would have used: '
          + off.map((s) => `<code>${escapeHtml(s)}</code>`).join(', ') + '.' : ''}
        </div>`;
    }
  }

  if (!valid.length) {
    h += `<div class="note bad"><b>No sample survived validation.</b>
      Try a higher temperature, more samples, or another target.</div>`;
  }

  valid.forEach((c, i) => {
    h += `<details ${i === 0 ? 'open' : ''}><summary>Proposal ${i + 1}
      — score ${c.score}${c.greedy ? ' · greedy' : ''}${
      c.n_samples > 1 ? ` · sampled ${c.n_samples}×` : ''}</summary>
      <div class="body">${flowHtml(c.graph, shelfSet)}
      <details class="src"><summary>canonical λproc document</summary>
        <div class="body"><pre>${escapeHtml(c.canonical)}</pre></div>
      </details></div></details>`;
  });

  for (const c of cands.filter((x) => !x.valid)) {
    h += `<details class="bad-doc"><summary>rejected — ${escapeHtml(c.reason)}
      </summary><div class="body"><pre>${escapeHtml(c.text)}</pre></div></details>`;
  }
  $('out').innerHTML = h;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

// ------------------------------------------------------------------ wiring
$('model').addEventListener('change', () => {
  $('shelfbox').hidden = $('model').value !== 'cond';
});
$('example').addEventListener('change', () => {
  $('target').value = $('example').value;
  depict();
});
$('target').addEventListener('change', depict);
$('go').addEventListener('click', generate);
start().catch((e) => { boot('Failed to start: ' + e); });
