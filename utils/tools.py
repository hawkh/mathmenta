"""
Utility tools for Math Mentor.
Provides safe calculation and helper functions.
"""
import ast
import operator
from typing import Any, Dict, Union


# Safe operators for mathematical expressions
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
}

ALLOWED_FUNCTIONS = {
    'abs': abs,
    'round': round,
    'min': min,
    'max': max,
    'sum': sum,
    'pow': pow,
}


class SafeCalculator:
    """
    Safe mathematical calculator using AST parsing.
    Prevents execution of arbitrary code while allowing mathematical expressions.
    """

    def evaluate(self, expression: str) -> Union[float, int, str]:
        """
        Safely evaluate a mathematical expression.

        Args:
            expression: Mathematical expression string

        Returns:
            Result of evaluation or error message
        """
        try:
            # Parse the expression
            tree = ast.parse(expression, mode='eval')
            return self._eval_node(tree.body)
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"

    def _eval_node(self, node: ast.AST) -> Union[float, int]:
        """Recursively evaluate AST node."""
        if isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.Constant):  # Python >= 3.8
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op_type = type(node.op)
            if op_type in ALLOWED_OPERATORS:
                return ALLOWED_OPERATORS[op_type](left, right)
            raise ValueError(f"Unsupported operator: {op_type}")
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op_type = type(node.op)
            if op_type in ALLOWED_OPERATORS:
                return ALLOWED_OPERATORS[op_type](operand)
            raise ValueError(f"Unsupported unary operator: {op_type}")
        elif isinstance(node, ast.Call):
            # Handle function calls like abs(), pow(), etc.
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in ALLOWED_FUNCTIONS:
                    args = [self._eval_node(arg) for arg in node.args]
                    return ALLOWED_FUNCTIONS[func_name](*args)
                raise ValueError(f"Unsupported function: {func_name}")
            raise ValueError("Unsupported call type")
        elif isinstance(node, ast.Attribute):
            # Handle math.pi, math.e, etc.
            if isinstance(node.value, ast.Name) and node.value.id == 'math':
                import math
                if hasattr(math, node.attr):
                    return getattr(math, node.attr)
                raise ValueError(f"Unknown math constant: {node.attr}")
            raise ValueError("Unsupported attribute access")
        else:
            raise ValueError(f"Unsupported expression type: {type(node)}")


# Singleton instance
_calculator_instance = None


def get_calculator() -> SafeCalculator:
    """Get or create the calculator singleton."""
    global _calculator_instance
    if _calculator_instance is None:
        _calculator_instance = SafeCalculator()
    return _calculator_instance


def safe_calculate(expression: str) -> Union[float, int, str]:
    """
    Convenience function to safely calculate a mathematical expression.

    Args:
        expression: Mathematical expression string

    Returns:
        Result of evaluation or error message
    """
    return get_calculator().evaluate(expression)


__all__ = [
    'SafeCalculator',
    'get_calculator',
    'safe_calculate'
]
