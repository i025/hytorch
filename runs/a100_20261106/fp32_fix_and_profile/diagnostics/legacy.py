"""Replay only the two original backward method bodies from the saved source."""
import ast, contextlib
from pathlib import Path
import hytorch.functions as F
@contextlib.contextmanager
def legacy(enabled=True):
    if not enabled:
        yield; return
    tree=ast.parse((Path(__file__).resolve().parents[1]/'raw/functions_before.py').read_text())
    saved={}
    for node in tree.body:
        if isinstance(node,ast.ClassDef) and node.name in ('MeanSquareError','Tanh'):
            method=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='backward')
            ns=dict(vars(F));exec(compile(ast.Module(body=[method],type_ignores=[]),'saved_pre_fix_backward','exec'),ns)
            cls=getattr(F,node.name);saved[cls]=cls.backward;cls.backward=ns['backward']
    try:yield
    finally:
        for cls,method in saved.items():cls.backward=method
