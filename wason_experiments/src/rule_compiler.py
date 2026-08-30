# src/rule_compiler.py
import ast
from typing import Callable
from .predicates_runtime import is_prime, is_cube, is_square

ALLOWED_FUNCS = {
    "is_prime": is_prime,
    "is_square": is_square,
    "is_cube": is_cube,
    "abs": abs,
    "all": all,
    "any": any,
    "len": len,
}

ALLOWED_VALUE_NAMES = {"a", "b", "c"}

ALLOWED_NODES = {
    # roots & containers
    ast.Expression, ast.Expr, ast.Module,
    # logic / arithmetic
    ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.Compare,
    ast.And, ast.Or, ast.Not,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.Is, ast.IsNot,
    # literals & names
    ast.Constant, ast.Name, ast.Load, ast.Set, ast.Tuple,
    # calls
    ast.Call,
    # comprehensions
    ast.GeneratorExp, ast.comprehension,
}

def _validate(node: ast.AST, gen_names=frozenset()):
    if type(node) not in ALLOWED_NODES:
        raise ValueError(f"Disallowed node: {type(node).__name__}")

    # allow a,b,c; generator-target names; and function names in ALLOWED_FUNCS
    if isinstance(node, ast.Name):
        if node.id not in ALLOWED_VALUE_NAMES and node.id not in gen_names and node.id not in ALLOWED_FUNCS:
            raise ValueError(f"Disallowed name: {node.id}")

    # comprehension: register the target name first, then validate iter/ifs
    if isinstance(node, ast.comprehension):
        # only simple targets like "for x in ..."
        if not isinstance(node.target, ast.Name):
            raise ValueError("Only simple generator targets supported")
        new_gen_names = gen_names | {node.target.id}
        _validate(node.iter, new_gen_names)
        for cond in node.ifs:
            _validate(cond, new_gen_names)
        return

    # generator exp: visit generators first (to collect names), then validate elt
    if isinstance(node, ast.GeneratorExp):
        gnames = gen_names
        for comp in node.generators:
            # each comp may introduce a new target
            if not isinstance(comp.target, ast.Name):
                raise ValueError("Only simple generator targets supported")
            gnames = gnames | {comp.target.id}
            _validate(comp, gnames)  # validates iter and ifs with growing gnames
        _validate(node.elt, gnames)
        return

    # default: recurse
    for child in ast.iter_child_nodes(node):
        _validate(child, gen_names)

def compile_expr(expr: str) -> Callable[[int,int,int], bool]:
    tree = ast.parse(expr, mode="eval")
    _validate(tree)
    code = compile(tree, "<rule_expr>", "eval")
    SAFE_GLOBALS = {"__builtins__": {}}
    SAFE_GLOBALS.update(ALLOWED_FUNCS)

    def pred(a: int, b: int, c: int) -> bool:
        # functions must be in globals so genexprs can resolve them
        return bool(eval(code, SAFE_GLOBALS, {"a": a, "b": b, "c": c}))
    return pred
