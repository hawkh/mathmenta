"""
Symbolic Mathematics Engine for JEE-level Problem Solving

This module provides comprehensive symbolic mathematics capabilities
using SymPy for accurate computation in calculus, algebra, linear algebra,
trigonometry, and more.
"""

import logging
from typing import Union, List, Dict, Any, Optional, Tuple
from functools import lru_cache
import hashlib

from sympy import (
    symbols, sympify, Symbol, 
    solve, solveset, Eq,
    diff, integrate, limit, series,
    Matrix, det, eye, zeros, ones,
    simplify, expand, factor, cancel, together, apart,
    sin, cos, tan, cot, sec, csc,
    asin, acos, atan,
    sinh, cosh, tanh,
    exp, log, sqrt, I, pi, E,
    latex, pretty, S,
    Integral, Derivative, Limit,
    Sum, Product,
    Rational, N,
    polarify, unpolarify,
    trigsimp, powsimp, radsimp, expand_trig,
    checkodesol, dsolve, Function,
    zoo, nan, oo
)
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import numpy as np

logger = logging.getLogger(__name__)


class SymbolicCalculator:
    """
    Production-grade symbolic mathematics engine for JEE-level problems.
    
    Supports:
    - Equation solving (linear, quadratic, polynomial, systems)
    - Calculus (differentiation, integration, limits)
    - Linear algebra (matrices, determinants, eigenvalues)
    - Trigonometry (simplification, identities, equations)
    - Complex numbers
    - Algebraic manipulation
    
    Examples:
        >>> calc = SymbolicCalculator()
        >>> calc.solve_equation("x**2 - 5*x + 6 = 0")
        ['2', '3']
        >>> calc.differentiate("sin(x**2) * exp(x)")
        '2 x e^{x} \\cos{\left(x^{2} \\right)} + e^{x} \\sin{\left(x^{2} \\right)}'
        >>> calc.integrate_expression("3*x**2 + 2*x + 1")
        'x^{3} + x^{2} + x'
    """
    
    # Cache size for repeated computations
    CACHE_SIZE = 1000
    
    def __init__(self):
        """Initialize the symbolic calculator with common symbols."""
        # Common symbols for quick access
        self.x, self.y, self.z = symbols('x y z', real=True)
        self.a, self.b, self.c = symbols('a b c', real=True)
        self.n, self.m = symbols('n m', integer=True)
        self.t, self.theta = symbols('t theta', real=True)
        
        # Common functions
        self.functions = {
            'sin': sin, 'cos': cos, 'tan': tan, 'cot': cot,
            'sec': sec, 'csc': csc,
            'asin': asin, 'acos': acos, 'atan': atan,
            'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
            'exp': exp, 'log': log, 'sqrt': sqrt,
            'abs': abs, 'factorial': lambda x: x  # Simplified
        }
        
        # Transformations for parsing
        self.transformations = (
            standard_transformations + 
            (implicit_multiplication_application,)
        )
        
        logger.info("SymbolicCalculator initialized")
    
    def _parse_expression(self, expr_str: str, local_dict: Optional[Dict] = None) -> Any:
        """
        Parse a mathematical expression from string or LaTeX.
        
        Args:
            expr_str: Mathematical expression as string or LaTeX
            local_dict: Optional dictionary of local symbol definitions
            
        Returns:
            Parsed SymPy expression
            
        Raises:
            ValueError: If expression cannot be parsed
        """
        if not expr_str:
            raise ValueError("Empty expression")
        
        expr_str = expr_str.strip()
        
        # Clean LaTeX formatting
        if '$' in expr_str:
            expr_str = expr_str.strip('$')
        
        # Handle common LaTeX patterns
        expr_str = self._clean_latex(expr_str)
        
        try:
            # Try parsing as LaTeX first
            if '\\frac' in expr_str or '\\sqrt' in expr_str or '^' in expr_str:
                parsed = parse_latex(expr_str)
            else:
                parsed = parse_expr(
                    expr_str,
                    local_dict=local_dict or self.functions,
                    transformations=self.transformations
                )
            return parsed
        except Exception as e:
            logger.warning(f"Failed to parse '{expr_str}': {e}")
            raise ValueError(f"Cannot parse expression: {expr_str}. Error: {str(e)}")
    
    def _clean_latex(self, latex_str: str) -> str:
        """
        Clean and normalize LaTeX string for parsing.
        
        Args:
            latex_str: Raw LaTeX string
            
        Returns:
            Cleaned LaTeX string
        """
        replacements = {
            r'\times': '*',
            r'\div': '/',
            r'\cdot': '*',
            r'\rightarrow': '->',
            r'\Rightarrow': '->',
            r'\left': '',
            r'\right': '',
            r'\ ': '',
            r'\,': '',
        }
        
        result = latex_str
        for old, new in replacements.items():
            result = result.replace(old, new)
        
        return result
    
    def _to_latex(self, expr: Any, mode: str = 'plain') -> str:
        """
        Convert SymPy expression to LaTeX string.
        
        Args:
            expr: SymPy expression
            mode: Output mode ('plain', 'display', 'inline')
            
        Returns:
            LaTeX string
        """
        try:
            latex_str = latex(expr)
            if mode == 'display':
                return f"\\[{latex_str}\\]"
            elif mode == 'inline':
                return f"${latex_str}$"
            return latex_str
        except Exception as e:
            logger.error(f"Failed to convert to LaTeX: {e}")
            return str(expr)
    
    def _hash_expression(self, expr: str) -> str:
        """Generate hash for caching."""
        return hashlib.md5(expr.encode()).hexdigest()
    
    # ==================== EQUATION SOLVING ====================
    
    def solve_equation(self, equation: str, variable: str = 'x') -> Dict[str, Any]:
        """
        Solve an algebraic equation.
        
        Args:
            equation: Equation string (e.g., "x**2 - 5*x + 6 = 0" or "x^2 + 2x + 1")
            variable: Variable to solve for
            
        Returns:
            Dictionary with solutions and metadata
            
        Examples:
            >>> calc.solve_equation("x**2 - 5*x + 6 = 0")
            {'solutions': ['2', '3'], 'method': 'algebraic', 'count': 2}
            
            >>> calc.solve_equation("2*x + 3 = 7")
            {'solutions': ['2'], 'method': 'linear', 'count': 1}
        """
        try:
            # Ensure variable is a string, not a Symbol object
            if hasattr(variable, 'name'):
                var_name = variable.name  # Extract name from Symbol
            else:
                var_name = str(variable)
            var = symbols(var_name)
            
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
            
            # Handle different solution types
            if isinstance(solutions, dict):
                # System of equations
                formatted_solutions = [str(v) for v in solutions.values()]
                method = 'system'
            elif isinstance(solutions, (list, tuple)):
                formatted_solutions = [self._to_latex(s) for s in solutions]
                
                # Determine method based on solution characteristics
                if len(solutions) == 1 and 'log' in str(solutions[0]).lower():
                    method = 'transcendental'
                elif len(solutions) == 2 and 'sqrt' in str(solutions[0]).lower():
                    method = 'quadratic'
                else:
                    method = 'algebraic'
            else:
                # Single solution
                formatted_solutions = [self._to_latex(solutions)]
                method = 'algebraic'
            
            # Remove duplicates while preserving order
            seen = set()
            unique_solutions = []
            for sol in formatted_solutions:
                if sol not in seen:
                    seen.add(sol)
                    unique_solutions.append(sol)
            
            return {
                'solutions': unique_solutions,
                'method': method,
                'count': len(unique_solutions),
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
        """
        Solve a system of equations.
        
        Args:
            equations: List of equation strings
            variables: List of variables to solve for
            
        Returns:
            Dictionary with solutions
            
        Examples:
            >>> calc.solve_system(["x + y = 5", "2*x - y = 1"], ["x", "y"])
            {'solutions': {'x': '2', 'y': '3'}, 'count': 2}
        """
        try:
            vars = [symbols(v) for v in variables]
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
            elif isinstance(solutions, (list, tuple)):
                if solutions and isinstance(solutions[0], (list, tuple)):
                    # Multiple solution sets
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
    
    # ==================== CALCULUS ====================
    
    def differentiate(self, expression: str, variable: str = 'x', order: int = 1) -> Dict[str, Any]:
        """
        Compute the derivative of an expression.
        
        Args:
            expression: Mathematical expression
            variable: Variable to differentiate with respect to
            order: Order of derivative (1 for first, 2 for second, etc.)
            
        Returns:
            Dictionary with derivative and metadata
            
        Examples:
            >>> calc.differentiate("sin(x**2) * exp(x)")
            {'derivative': '2 x e^{x} \\cos{\left(x^{2} \\right)} + e^{x} \\sin{\left(x^{2} \\right)}', 
             'simplified': 'e^{x} \\left(2 x \\cos{\left(x^{2} \\right)} + \\sin{\left(x^{2} \\right)}\\right)'}
             
            >>> calc.differentiate("x**3 + 2*x**2 + x", order=2)
            {'derivative': '6 x + 4', ...}
        """
        try:
            var = symbols(variable)
            expr = self._parse_expression(expression)
            
            # Compute derivative
            result = diff(expr, var, order)
            
            # Simplify
            simplified = simplify(result)
            
            # Check if result contains unevaluated derivative
            if result.has(Derivative):
                return {
                    'derivative': self._to_latex(result),
                    'simplified': self._to_latex(result),
                    'error': 'Could not evaluate derivative symbolically',
                    'method': 'failed'
                }
            
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
        """
        Compute the integral of an expression.
        
        Args:
            expression: Mathematical expression
            variable: Variable to integrate with respect to
            lower: Lower bound for definite integral (optional)
            upper: Upper bound for definite integral (optional)
            
        Returns:
            Dictionary with integral and metadata
            
        Examples:
            >>> calc.integrate_expression("3*x**2 + 2*x + 1")
            {'integral': 'x^{3} + x^{2} + x', 'type': 'indefinite'}
            
            >>> calc.integrate_expression("x**2", lower=0, upper=1)
            {'integral': '1/3', 'type': 'definite', 'value': 0.333...}
        """
        try:
            # Ensure variable is a string
            var_name = variable.name if hasattr(variable, 'name') else str(variable)
            var = symbols(var_name)
            expr = self._parse_expression(expression)
            
            if lower is not None and upper is not None:
                # Definite integral
                result = integrate(expr, (var, lower, upper))
                integral_type = 'definite'
                
                # Try to get numerical value
                try:
                    numerical_value = float(result.evalf())
                except:
                    numerical_value = None
                
                return {
                    'integral': self._to_latex(result),
                    'type': integral_type,
                    'bounds': (lower, upper),
                    'numerical_value': numerical_value,
                    'original': expression,
                    'method': 'symbolic'
                }
            else:
                # Indefinite integral
                result = integrate(expr, var)
                
                # Check if integration failed
                if result.has(Integral):
                    return {
                        'integral': self._to_latex(result),
                        'type': 'indefinite',
                        'warning': 'Could not evaluate integral in closed form',
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
        """
        Compute the limit of an expression.
        
        Args:
            expression: Mathematical expression
            variable: Variable
            point: Point to approach (can be 'oo' for infinity)
            
        Returns:
            Dictionary with limit value and metadata
            
        Examples:
            >>> calc.compute_limit("sin(x)/x", point=0)
            {'limit': '1', 'point': 0}
            
            >>> calc.compute_limit("1/x", point='oo')
            {'limit': '0', 'point': 'oo'}
        """
        try:
            var = symbols(variable)
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
    
    # ==================== LINEAR ALGEBRA ====================
    
    def matrix_operation(self, matrix_str: str, operation: str,
                        second_matrix: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform matrix operations.
        
        Args:
            matrix_str: Matrix as string "[[1,2],[3,4]]"
            operation: Operation type ('det', 'inverse', 'eigenvalues', 
                      'eigenvectors', 'rank', 'transpose', 'rref')
            second_matrix: Second matrix for binary operations
            
        Returns:
            Dictionary with result
            
        Examples:
            >>> calc.matrix_operation("[[1,2],[3,4]]", "det")
            {'result': '-2', 'operation': 'det'}
            
            >>> calc.matrix_operation("[[1,2],[3,4]]", "eigenvalues")
            {'result': ['(5/2 - sqrt(33)/2)', '(5/2 + sqrt(33)/2)'], ...}
        """
        try:
            import ast
            import sympy
            
            # Parse matrix
            matrix_data = ast.literal_eval(matrix_str)
            M = Matrix(matrix_data)
            
            if operation == 'det' or operation == 'determinant':
                result = det(M)
                return {
                    'result': self._to_latex(result),
                    'operation': 'determinant',
                    'matrix_size': M.shape
                }
            
            elif operation == 'inverse' or operation == 'inv':
                if det(M) == 0:
                    return {
                        'error': 'Matrix is singular, no inverse exists',
                        'operation': 'inverse'
                    }
                result = M.inv()
                return {
                    'result': self._to_latex(result),
                    'operation': 'inverse',
                    'matrix_size': M.shape
                }
            
            elif operation == 'eigenvalues' or operation == 'eigs':
                eigs = M.eigenvals()
                return {
                    'result': [self._to_latex(e) for e in eigs.keys()],
                    'multiplicities': {self._to_latex(k): v for k, v in eigs.items()},
                    'operation': 'eigenvalues',
                    'count': len(eigs)
                }
            
            elif operation == 'eigenvectors':
                eigs = M.eigenvects()
                return {
                    'result': [
                        {
                            'eigenvalue': self._to_latex(eig[0]),
                            'multiplicity': eig[1],
                            'vectors': [self._to_latex(v) for v in eig[2]]
                        }
                        for eig in eigs
                    ],
                    'operation': 'eigenvectors',
                    'count': len(eigs)
                }
            
            elif operation == 'rank':
                result = M.rank()
                return {
                    'result': int(result),
                    'operation': 'rank',
                    'matrix_size': M.shape
                }
            
            elif operation == 'transpose' or operation == 'T':
                result = M.T
                return {
                    'result': self._to_latex(result),
                    'operation': 'transpose',
                    'matrix_size': M.shape
                }
            
            elif operation == 'rref':
                result, pivot_cols = M.rref()
                return {
                    'result': self._to_latex(result),
                    'pivot_columns': list(pivot_cols),
                    'operation': 'rref',
                    'matrix_size': M.shape
                }
            
            elif operation == 'multiply' or operation == 'matmul':
                if second_matrix is None:
                    return {'error': 'Second matrix required for multiplication'}
                M2 = Matrix(ast.literal_eval(second_matrix))
                result = M * M2
                return {
                    'result': self._to_latex(result),
                    'operation': 'multiplication',
                    'matrix_size': result.shape
                }
            
            else:
                return {'error': f'Unknown operation: {operation}'}
                
        except Exception as e:
            logger.exception(f"Error in matrix operation: {e}")
            return {
                'error': str(e),
                'operation': operation
            }
    
    # ==================== TRIGONOMETRY ====================
    
    def simplify_trig(self, expression: str) -> Dict[str, Any]:
        """
        Simplify a trigonometric expression.
        
        Args:
            expression: Trigonometric expression
            
        Returns:
            Dictionary with simplified form and steps
            
        Examples:
            >>> calc.simplify_trig("sin(x)**2 + cos(x)**2")
            {'simplified': '1', 'identity': 'Pythagorean'}
            
            >>> calc.simplify_trig("sin(2*x)")
            {'simplified': '2 sin(x) cos(x)', 'identity': 'Double angle'}
        """
        try:
            expr = self._parse_expression(expression)
            
            # Apply trigonometric simplification
            simplified = trigsimp(expr)
            
            # Try other simplifications
            alternatives = {
                'expanded': self._to_latex(expand_trig(expr)),
                'factored': self._to_latex(factor(expr))
            }
            
            return {
                'simplified': self._to_latex(simplified),
                'original': expression,
                'alternatives': alternatives,
                'method': 'trigsimp'
            }
            
        except Exception as e:
            logger.exception(f"Error simplifying trig: {e}")
            return {
                'simplified': '',
                'error': str(e),
                'method': 'failed'
            }
    
    def solve_trig_equation(self, equation: str, variable: str = 'x') -> Dict[str, Any]:
        """
        Solve a trigonometric equation.
        
        Args:
            equation: Trigonometric equation
            variable: Variable to solve for
            
        Returns:
            Dictionary with solutions
            
        Examples:
            >>> calc.solve_trig_equation("sin(x) = 0.5")
            {'solutions': ['pi/6', '5*pi/6'], 'general': 'pi/6 + 2*pi*n, 5*pi/6 + 2*pi*n'}
        """
        try:
            var = symbols(variable)
            
            if '=' in equation:
                left, right = equation.split('=', 1)
                eq = Eq(
                    self._parse_expression(left.strip()),
                    self._parse_expression(right.strip())
                )
            else:
                eq = Eq(self._parse_expression(equation), 0)
            
            # Use solveset for trigonometric equations
            solutions = solveset(eq, var)
            
            return {
                'solutions': str(solutions),
                'latex': self._to_latex(solutions),
                'original': equation,
                'method': 'trigonometric'
            }
            
        except Exception as e:
            logger.exception(f"Error solving trig equation: {e}")
            return {
                'solutions': '',
                'error': str(e),
                'method': 'failed'
            }
    
    # ==================== VERIFICATION ====================
    
    def verify_solution(self, equation: str, solution: str, 
                       variable: str = 'x') -> Dict[str, Any]:
        """
        Verify a solution by substitution.
        
        Args:
            equation: Original equation
            solution: Proposed solution
            variable: Variable
            
        Returns:
            Dictionary with verification result
            
        Examples:
            >>> calc.verify_solution("x**2 - 5*x + 6 = 0", "2")
            {'is_correct': True, 'verification': 'LHS = RHS = 0'}
        """
        try:
            var = symbols(variable)
            
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
                # Try as a number
                sol_value = float(solution)
            
            # Substitute
            lhs_substituted = lhs.subs(var, sol_value)
            rhs_substituted = rhs.subs(var, sol_value)
            
            # Check equality
            difference = simplify(lhs_substituted - rhs_substituted)
            is_correct = difference == 0
            
            return {
                'is_correct': is_correct,
                'lhs_value': self._to_latex(lhs_substituted),
                'rhs_value': self._to_latex(rhs_substituted),
                'difference': self._to_latex(difference),
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
        """
        Verify a derivative by numerical checking.
        
        Args:
            original: Original function
            derivative: Claimed derivative
            variable: Variable
            
        Returns:
            Dictionary with verification result
        """
        try:
            var = symbols(variable)
            f = self._parse_expression(original)
            g = self._parse_expression(derivative)
            
            # Compute actual derivative
            actual_derivative = diff(f, var)
            
            # Check if claimed derivative matches
            difference = simplify(g - actual_derivative)
            is_correct = difference == 0
            
            return {
                'is_correct': is_correct,
                'actual_derivative': self._to_latex(actual_derivative),
                'difference': self._to_latex(difference),
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
        """
        Verify an antiderivative by differentiation.
        
        Args:
            integrand: Function to integrate
            antiderivative: Claimed antiderivative
            variable: Variable
            
        Returns:
            Dictionary with verification result
        """
        try:
            var = symbols(variable)
            f = self._parse_expression(integrand)
            F = self._parse_expression(antiderivative)
            
            # Differentiate claimed antiderivative
            computed_derivative = diff(F, var)
            
            # Check if it matches original integrand
            difference = simplify(computed_derivative - f)
            is_correct = difference == 0
            
            return {
                'is_correct': is_correct,
                'computed_derivative': self._to_latex(computed_derivative),
                'difference': self._to_latex(difference),
                'method': 'differentiation_check'
            }
            
        except Exception as e:
            logger.exception(f"Error verifying integral: {e}")
            return {
                'is_correct': False,
                'error': str(e),
                'method': 'failed'
            }
    
    # ==================== UTILITY METHODS ====================
    
    def evaluate_expression(self, expression: str, 
                           substitutions: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Evaluate an expression numerically.
        
        Args:
            expression: Mathematical expression
            substitutions: Dictionary of variable substitutions
            
        Returns:
            Dictionary with numerical result
            
        Examples:
            >>> calc.evaluate_expression("x**2 + 2*x + 1", {"x": 3})
            {'result': 16.0, 'expression': 'x**2 + 2*x + 1'}
        """
        try:
            expr = self._parse_expression(expression)
            
            if substitutions:
                for var, value in substitutions.items():
                    expr = expr.subs(symbols(var), value)
            
            # Evaluate numerically
            result = expr.evalf()
            
            return {
                'result': float(result),
                'exact': self._to_latex(expr),
                'expression': expression
            }
            
        except Exception as e:
            logger.exception(f"Error evaluating: {e}")
            return {
                'result': None,
                'error': str(e)
            }
    
    def get_step_by_step(self, operation: str, expression: str, 
                        **kwargs) -> Dict[str, Any]:
        """
        Get step-by-step solution for an operation.
        
        Note: Full step-by-step requires SymPy's internal steps module
        which is experimental. This provides a basic framework.
        
        Args:
            operation: Operation type ('differentiate', 'integrate', 'solve')
            expression: Expression to operate on
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with steps (if available) or result
        """
        if operation == 'differentiate':
            result = self.differentiate(expression, **kwargs)
            return {
                'operation': 'differentiation',
                'input': expression,
                'result': result.get('derivative', ''),
                'simplified': result.get('simplified', ''),
                'steps': ['Step-by-step not yet available', 
                         f'Derivative: {result.get("derivative", "")}']
            }
        
        elif operation == 'integrate':
            result = self.integrate_expression(expression, **kwargs)
            return {
                'operation': 'integration',
                'input': expression,
                'result': result.get('integral', ''),
                'steps': ['Step-by-step not yet available',
                         f'Integral: {result.get("integral", "")}']
            }
        
        elif operation == 'solve':
            result = self.solve_equation(expression, **kwargs)
            return {
                'operation': 'equation solving',
                'input': expression,
                'solutions': result.get('solutions', []),
                'method': result.get('method', ''),
                'steps': ['Step-by-step not yet available',
                         f'Solutions: {result.get("solutions", [])}']
            }
        
        return {'error': f'Unknown operation: {operation}'}


# Convenience functions for direct use
def get_calculator() -> SymbolicCalculator:
    """Get a configured SymbolicCalculator instance."""
    return SymbolicCalculator()


# Module-level convenience functions
_calc_instance = None

def _get_calc():
    global _calc_instance
    if _calc_instance is None:
        _calc_instance = SymbolicCalculator()
    return _calc_instance


def solve(equation: str, variable: str = 'x') -> List[str]:
    """Solve an equation."""
    return _get_calc().solve_equation(equation, variable)['solutions']


def diff_expr(expression: str, variable: str = 'x') -> str:
    """Differentiate an expression."""
    return _get_calc().differentiate(expression, variable)['simplified']


def integrate(expression: str, variable: str = 'x') -> str:
    """Integrate an expression."""
    return _get_calc().integrate_expression(expression, variable)['integral']
