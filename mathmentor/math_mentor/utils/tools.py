"""
utils/tools.py
Tools available to the Solver Agent:
- Python calculator (safe eval of mathematical expressions)
- Symbolic math via Python's math module
"""
import math
import re
from typing import Any

# Safe builtins for calculator
_SAFE_GLOBALS = {
    "__builtins__": {},
    # Math constants
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
    "nan": math.nan,
    # Math functions
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "cbrt": lambda x: x ** (1 / 3) if x >= 0 else -((-x) ** (1 / 3)),
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "pow": math.pow,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "comb": math.comb,    # C(n, k)
    "perm": math.perm,    # P(n, k)
    "degrees": math.degrees,
    "radians": math.radians,
    # Python builtins that are safe
    "range": range,
    "len": len,
    "list": list,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "True": True,
    "False": False,
    "None": None,
}


def safe_calculate(expression: str) -> dict:
    """
    Safely evaluate a mathematical expression string.

    Args:
        expression: A Python-style math expression, e.g. "sqrt(16) + 2**3"

    Returns:
        {"result": <value>, "success": bool, "error": str}
    """
    # Normalise common notation to Python
    expr = expression.strip()
    expr = re.sub(r"\^", "**", expr)          # x^2 → x**2
    expr = re.sub(r"√\((.+?)\)", r"sqrt(\1)", expr)   # √(x) → sqrt(x)
    expr = re.sub(r"×", "*", expr)
    expr = re.sub(r"÷", "/", expr)

    try:
        result = eval(expr, _SAFE_GLOBALS, {})  # noqa: S307 (safe globals only)
        return {"result": result, "success": True, "error": ""}
    except ZeroDivisionError:
        return {"result": None, "success": False, "error": "Division by zero"}
    except (SyntaxError, NameError, TypeError, ValueError) as exc:
        return {"result": None, "success": False, "error": str(exc)}


def evaluate_expression_steps(steps: list[str]) -> list[dict]:
    """
    Evaluate a list of expression strings, returning result for each.
    Useful for verifying intermediate calculation steps.
    """
    return [safe_calculate(s) for s in steps]


def format_number(value: Any, precision: int = 6) -> str:
    """Format a number for display (round floats, keep ints exact)."""
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return f"{value:.{precision}g}"
    return str(value)
