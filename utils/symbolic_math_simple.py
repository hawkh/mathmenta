"""
Symbolic Mathematics Engine for JEE-level Problem Solving
Simplified version without caching issues.
"""

import logging
from typing import Union, List, Dict, Any, Optional

from sympy import (
    symbols, sympify, Symbol, 
    solve, solveset, Eq,
    diff, integrate, limit,
    Matrix, det,
    simplify, expand, factor,
    sin, cos, tan,
    exp, log, sqrt, I, pi,
    latex,
    Integral, Derivative,
    trigsimp, expand_trig,
    checkodesol, dsolve, Function,
    oo
)
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

logger = logging.getLogger(__name__)


class SymbolicCalculator:
    """
    Symbolic mathematics engine for JEE-level problems.
    """
    
    def __init__(self):
        """Initialize the symbolic calculator."""
        # Transformations for parsing
        self.transformations = (
            standard_transformations + 
            (implicit_multiplication_application,)
        )
        logger.info("SymbolicCalculator initialized")
    
    def _parse_expression(self, expr_str: str) -> Any:
        """Parse a mathematical expression from string."""
        if not expr_str:
            raise ValueError("Empty expression")
        
        expr_str = str(expr_str).strip()
        
        # Clean LaTeX formatting
        if '$' in expr_str:
            expr_str = expr_str.strip('$')
        
        try:
            # Try parsing as LaTeX first
            if '\\frac' in expr_str or '\\sqrt' in expr_str:
                parsed = parse_latex(expr_str)
            else:
                parsed = parse_expr(
                    expr_str,
                    transformations=self.transformations
                )
            return parsed
        except Exception as e:
            logger.warning(f"Failed to parse '{expr_str}': {e}")
            raise ValueError(f"Cannot parse expression: {expr_str}")
    
    def _to_latex(self, expr: Any) -> str:
        """Convert SymPy expression to LaTeX string."""
        try:
            return latex(expr)
        except Exception as e:
            logger.error(f"Failed to convert to LaTeX: {e}")
            return str(expr)
    
    def solve_equation(self, equation: str, variable: str = 'x') -> Dict[str, Any]:
        """
        Solve an algebraic equation.
        
        Args:
            equation: Equation string (e.g., "x**2 - 5*x + 6 = 0")
            variable: Variable to solve for
            
        Returns:
            Dictionary with solutions
        """
        try:
            var = Symbol(variable)
            
            # Parse equation
            if '=' in equation:
                left, right = equation.split('=', 1)
                left_expr = self._parse_expression(left.strip())
                right_expr = self._parse_expression(right.strip())
                eq = Eq(left_expr, right_expr)
            else:
                expr = self._parse_expression(equation)
                eq = Eq(expr, 0)
            
            # Solve
            solutions = solve(eq, var)
            
            # Format solutions
            if isinstance(solutions, (list, tuple)):
                formatted = [self._to_latex(s) for s in solutions]
            elif isinstance(solutions, dict):
                formatted = [self._to_latex(v) for v in solutions.values()]
            else:
                formatted = [self._to_latex(solutions)]
            
            # Remove duplicates
            unique = list(dict.fromkeys(formatted))
            
            return {
                'solutions': unique,
                'method': 'algebraic',
                'count': len(unique),
                'variable': variable,
                'original': equation
            }
            
        except Exception as e:
            logger.exception(f"Error solving equation: {e}")
            return {
                'solutions': [],
                'error': str(e),
                'method': 'failed',
                'count': 0
            }
    
    def solve_system(self, equations: List[str], variables: List[str]) -> Dict[str, Any]:
        """Solve a system of equations."""
        try:
            vars = [Symbol(v) for v in variables]
            eqs = []
            
            for eq_str in equations:
                if '=' in eq_str:
                    left, right = eq_str.split('=', 1)
                    eqs.append(Eq(
                        self._parse_expression(left.strip()),
                        self._parse_expression(right.strip())
                    ))
                else:
                    eqs.append(Eq(self._parse_expression(eq_str), 0))
            
            solutions = solve(eqs, vars)
            
            if isinstance(solutions, dict):
                return {
                    'solutions': {k: self._to_latex(v) for k, v in solutions.items()},
                    'count': len(solutions)
                }
            elif isinstance(solutions, (list, tuple)) and solutions:
                if isinstance(solutions[0], (list, tuple)):
                    return {
                        'solutions': [
                            {str(vars[i]): self._to_latex(val) for i, val in enumerate(sol)}
                            for sol in solutions
                        ],
                        'count': len(solutions)
                    }
                else:
                    return {
                        'solutions': {str(vars[i]): self._to_latex(val) for i, val in enumerate(solutions)},
                        'count': len(variables)
                    }
            
            return {'solutions': {}, 'count': 0}
            
        except Exception as e:
            logger.exception(f"Error solving system: {e}")
            return {'solutions': {}, 'error': str(e), 'count': 0}
    
    def differentiate(self, expression: str, variable: str = 'x', order: int = 1) -> Dict[str, Any]:
        """Compute the derivative of an expression."""
        try:
            var = Symbol(variable)
            expr = self._parse_expression(expression)
            
            # Compute derivative
            result = diff(expr, var, order)
            simplified = simplify(result)
            
            return {
                'derivative': self._to_latex(result),
                'simplified': self._to_latex(simplified),
                'variable': variable,
                'order': order,
                'original': expression,
                'method': 'symbolic'
            }
            
        except Exception as e:
            logger.exception(f"Error differentiating: {e}")
            return {
                'derivative': '',
                'error': str(e),
                'method': 'failed'
            }
    
    def integrate_expression(self, expression: str, variable: str = 'x',
                            lower: Optional[float] = None, 
                            upper: Optional[float] = None) -> Dict[str, Any]:
        """Compute the integral of an expression."""
        try:
            var = Symbol(variable)
            expr = self._parse_expression(expression)
            
            if lower is not None and upper is not None:
                # Definite integral
                result = integrate(expr, (var, lower, upper))
                
                try:
                    numerical_value = float(result.evalf())
                except:
                    numerical_value = None
                
                return {
                    'integral': self._to_latex(result),
                    'type': 'definite',
                    'bounds': (lower, upper),
                    'numerical_value': numerical_value,
                    'original': expression,
                    'method': 'symbolic'
                }
            else:
                # Indefinite integral
                result = integrate(expr, var)
                
                if result.has(Integral):
                    return {
                        'integral': self._to_latex(result),
                        'type': 'indefinite',
                        'warning': 'Could not evaluate in closed form',
                        'original': expression,
                        'method': 'partial'
                    }
                
                return {
                    'integral': self._to_latex(result),
                    'type': 'indefinite',
                    'original': expression,
                    'method': 'symbolic'
                }
                
        except Exception as e:
            logger.exception(f"Error integrating: {e}")
            return {
                'integral': '',
                'error': str(e),
                'method': 'failed'
            }
    
    def compute_limit(self, expression: str, variable: str = 'x', 
                      point: Union[float, str] = 0) -> Dict[str, Any]:
        """Compute the limit of an expression."""
        try:
            var = Symbol(variable)
            expr = self._parse_expression(expression)
            
            # Handle special points
            if point == 'oo' or point == 'inf' or point == 'infinity':
                pt = oo
            elif point == '-oo' or point == '-inf':
                pt = -oo
            else:
                pt = point
            
            result = limit(expr, var, pt)
            
            return {
                'limit': self._to_latex(result),
                'point': point,
                'variable': variable,
                'original': expression,
                'method': 'symbolic'
            }
            
        except Exception as e:
            logger.exception(f"Error computing limit: {e}")
            return {
                'limit': '',
                'error': str(e),
                'method': 'failed'
            }
    
    def matrix_operation(self, matrix_str: str, operation: str,
                        second_matrix: Optional[str] = None) -> Dict[str, Any]:
        """Perform matrix operations."""
        try:
            import ast
            
            # Parse matrix
            matrix_data = ast.literal_eval(matrix_str)
            M = Matrix(matrix_data)
            
            if operation in ['det', 'determinant']:
                return {
                    'result': self._to_latex(det(M)),
                    'operation': 'determinant',
                    'matrix_size': M.shape
                }
            
            elif operation in ['inverse', 'inv']:
                if det(M) == 0:
                    return {'error': 'Matrix is singular', 'operation': 'inverse'}
                return {
                    'result': self._to_latex(M.inv()),
                    'operation': 'inverse',
                    'matrix_size': M.shape
                }
            
            elif operation == 'eigenvalues':
                eigs = M.eigenvals()
                return {
                    'result': [self._to_latex(e) for e in eigs.keys()],
                    'multiplicities': {self._to_latex(k): v for k, v in eigs.items()},
                    'operation': 'eigenvalues',
                    'count': len(eigs)
                }
            
            elif operation == 'rank':
                return {
                    'result': int(M.rank()),
                    'operation': 'rank',
                    'matrix_size': M.shape
                }
            
            elif operation in ['transpose', 'T']:
                return {
                    'result': self._to_latex(M.T),
                    'operation': 'transpose',
                    'matrix_size': M.shape
                }
            
            else:
                return {'error': f'Unknown operation: {operation}'}
                
        except Exception as e:
            logger.exception(f"Error in matrix operation: {e}")
            return {'error': str(e), 'operation': operation}
    
    def simplify_trig(self, expression: str) -> Dict[str, Any]:
        """Simplify a trigonometric expression."""
        try:
            expr = self._parse_expression(expression)
            simplified = trigsimp(expr)
            
            return {
                'simplified': self._to_latex(simplified),
                'original': expression,
                'method': 'trigsimp'
            }
            
        except Exception as e:
            logger.exception(f"Error simplifying trig: {e}")
            return {
                'simplified': '',
                'error': str(e),
                'method': 'failed'
            }
    
    def verify_solution(self, equation: str, solution: str, 
                       variable: str = 'x') -> Dict[str, Any]:
        """Verify a solution by substitution."""
        try:
            var = Symbol(variable)
            
            # Parse equation
            if '=' in equation:
                left, right = equation.split('=', 1)
                lhs = self._parse_expression(left.strip())
                rhs = self._parse_expression(right.strip())
            else:
                lhs = self._parse_expression(equation)
                rhs = 0
            
            # Parse solution
            try:
                sol_value = self._parse_expression(solution)
            except:
                sol_value = float(solution)
            
            # Substitute and check
            lhs_sub = lhs.subs(var, sol_value)
            rhs_sub = rhs.subs(var, sol_value)
            difference = simplify(lhs_sub - rhs_sub)
            is_correct = difference == 0
            
            return {
                'is_correct': is_correct,
                'lhs_value': self._to_latex(lhs_sub),
                'rhs_value': self._to_latex(rhs_sub),
                'method': 'substitution'
            }
            
        except Exception as e:
            logger.exception(f"Error verifying solution: {e}")
            return {
                'is_correct': False,
                'error': str(e),
                'method': 'failed'
            }
    
    def verify_derivative(self, original: str, derivative: str, 
                         variable: str = 'x') -> Dict[str, Any]:
        """Verify a derivative by symbolic comparison."""
        try:
            var = Symbol(variable)
            f = self._parse_expression(original)
            g = self._parse_expression(derivative)
            
            actual = diff(f, var)
            difference = simplify(g - actual)
            is_correct = difference == 0
            
            return {
                'is_correct': is_correct,
                'actual_derivative': self._to_latex(actual),
                'method': 'symbolic_comparison'
            }
            
        except Exception as e:
            logger.exception(f"Error verifying derivative: {e}")
            return {
                'is_correct': False,
                'error': str(e),
                'method': 'failed'
            }
    
    def verify_integral(self, integrand: str, antiderivative: str,
                       variable: str = 'x') -> Dict[str, Any]:
        """Verify an antiderivative by differentiation."""
        try:
            var = Symbol(variable)
            f = self._parse_expression(integrand)
            F = self._parse_expression(antiderivative)
            
            computed = diff(F, var)
            difference = simplify(computed - f)
            is_correct = difference == 0
            
            return {
                'is_correct': is_correct,
                'computed_derivative': self._to_latex(computed),
                'method': 'differentiation_check'
            }
            
        except Exception as e:
            logger.exception(f"Error verifying integral: {e}")
            return {
                'is_correct': False,
                'error': str(e),
                'method': 'failed'
            }


def get_calculator() -> SymbolicCalculator:
    """Get a configured SymbolicCalculator instance."""
    return SymbolicCalculator()
