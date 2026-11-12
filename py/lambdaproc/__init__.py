"""λproc: an S-expression language for process description.

Processes (recipes, industrial flowsheets) are written as compositions of
unit operations bound to a machine database, with lambda-based sub-process
macros and named multi-output selectors.
"""

from .core import (
    App,
    Def,
    Document,
    Lambda,
    LambdaProcError,
    MatBinding,
    MatDecl,
    ParseError,
    Ref,
    UseBinding,
    UseDecl,
    canonical,
    load_machines,
    parse,
    to_dag,
    validate,
)

__version__ = "0.1.0"

__all__ = [
    "App", "Def", "Document", "Lambda", "LambdaProcError", "MatBinding",
    "MatDecl", "ParseError", "Ref", "UseBinding", "UseDecl",
    "canonical", "load_machines", "parse", "to_dag", "validate",
]
