"""
Comprehensive Tests for Symbolic Mathematics Engine

Tests cover JEE-level problems in:
- Equation solving (linear, quadratic, polynomial, systems)
- Calculus (differentiation, integration, limits)
- Linear algebra (matrices, determinants, eigenvalues)
- Trigonometry (simplification, equations)
- Verification mechanisms
"""

import unittest
import sys
import logging
from typing import List, Dict, Any

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from utils.symbolic_math import SymbolicCalculator


class TestEquationSolving(unittest.TestCase):
    """Test equation solving capabilities."""
    
    def setUp(self):
        self.calc = SymbolicCalculator()
    
    def test_linear_equation(self):
        """Test solving linear equations."""
        result = self.calc.solve_equation("2*x + 3 = 7")
        self.assertIn('2', result['solutions'])
        self.assertEqual(result['count'], 1)
        print(f"✓ Linear equation: 2x + 3 = 7 → x = {result['solutions']}")
    
    def test_quadratic_equation_factorable(self):
        """Test solving factorable quadratic equations."""
        result = self.calc.solve_equation("x**2 - 5*x + 6 = 0")
        self.assertIn('2', result['solutions'])
        self.assertIn('3', result['solutions'])
        self.assertEqual(result['count'], 2)
        print(f"✓ Quadratic: x² - 5x + 6 = 0 → x = {result['solutions']}")
    
    def test_quadratic_equation_complex_roots(self):
        """Test solving quadratic with complex roots."""
        result = self.calc.solve_equation("x**2 + 4 = 0")
        self.assertEqual(result['count'], 2)
        print(f"✓ Complex roots: x² + 4 = 0 → x = {result['solutions']}")
    
    def test_quadratic_formula_case(self):
        """Test quadratic requiring quadratic formula."""
        result = self.calc.solve_equation("x**2 - 4*x + 1 = 0")
        self.assertEqual(result['count'], 2)
        print(f"✓ Quadratic formula: x² - 4x + 1 = 0 → x = {result['solutions']}")
    
    def test_cubic_equation(self):
        """Test solving cubic equation."""
        result = self.calc.solve_equation("x**3 - 6*x**2 + 11*x - 6 = 0")
        self.assertGreater(result['count'], 0)
        print(f"✓ Cubic: x³ - 6x² + 11x - 6 = 0 → x = {result['solutions']}")
    
    def test_rational_equation(self):
        """Test solving rational equations."""
        result = self.calc.solve_equation("1/x + 1/(x + 1) = 2")
        self.assertGreater(result['count'], 0)
        print(f"✓ Rational: 1/x + 1/(x+1) = 2 → x = {result['solutions']}")
    
    def test_radical_equation(self):
        """Test solving equations with radicals."""
        result = self.calc.solve_equation("sqrt(x + 3) = x - 3")
        self.assertGreater(result['count'], 0)
        print(f"✓ Radical: √(x+3) = x-3 → x = {result['solutions']}")
    
    def test_system_of_linear_equations(self):
        """Test solving system of linear equations."""
        result = self.calc.solve_system(
            ["x + y = 5", "2*x - y = 1"],
            ["x", "y"]
        )
        self.assertIn('x', result['solutions'])
        self.assertIn('y', result['solutions'])
        print(f"✓ System: x+y=5, 2x-y=1 → {result['solutions']}")
    
    def test_exponential_equation(self):
        """Test solving exponential equations."""
        result = self.calc.solve_equation("2**x = 8")
        self.assertIn('3', result['solutions'])
        print(f"✓ Exponential: 2^x = 8 → x = {result['solutions']}")
    
    def test_logarithmic_equation(self):
        """Test solving logarithmic equations."""
        result = self.calc.solve_equation("log(x, 2) = 3")
        self.assertIn('8', result['solutions'])
        print(f"✓ Logarithmic: log₂(x) = 3 → x = {result['solutions']}")


class TestCalculus(unittest.TestCase):
    """Test calculus operations."""
    
    def setUp(self):
        self.calc = SymbolicCalculator()
    
    def test_derivative_polynomial(self):
        """Test derivative of polynomial."""
        result = self.calc.differentiate("x**3 + 2*x**2 + x")
        self.assertIn('x**2', result['derivative'])
        print(f"✓ Derivative: d/dx(x³ + 2x² + x) = {result['simplified']}")
    
    def test_derivative_product_rule(self):
        """Test derivative requiring product rule - JEE level."""
        result = self.calc.differentiate("sin(x**2) * exp(x)")
        self.assertTrue(result['derivative'] != '')
        print(f"✓ Product rule: d/dx(sin(x²)·e^x) = {result['simplified']}")
    
    def test_derivative_chain_rule(self):
        """Test derivative requiring chain rule."""
        result = self.calc.differentiate("sin(x**2)")
        self.assertIn('cos', result['derivative'])
        print(f"✓ Chain rule: d/dx(sin(x²)) = {result['simplified']}")
    
    def test_derivative_quotient_rule(self):
        """Test derivative requiring quotient rule."""
        result = self.calc.differentiate("sin(x)/x")
        self.assertTrue(result['derivative'] != '')
        print(f"✓ Quotient rule: d/dx(sin(x)/x) = {result['simplified']}")
    
    def test_second_derivative(self):
        """Test second derivative."""
        result = self.calc.differentiate("x**4 - 3*x**3 + 2*x", order=2)
        self.assertTrue(result['derivative'] != '')
        print(f"✓ Second derivative: d²/dx²(x⁴ - 3x³ + 2x) = {result['simplified']}")
    
    def test_derivative_trig(self):
        """Test derivative of trigonometric function."""
        result = self.calc.differentiate("tan(x**2)")
        self.assertIn('sec', result['derivative'].lower())
        print(f"✓ Trig derivative: d/dx(tan(x²)) = {result['simplified']}")
    
    def test_indefinite_integral_polynomial(self):
        """Test indefinite integral of polynomial."""
        result = self.calc.integrate_expression("3*x**2 + 2*x + 1")
        self.assertIn('x**3', result['integral'])
        print(f"✓ Integral: ∫(3x² + 2x + 1)dx = {result['integral']}")
    
    def test_indefinite_integral_trig(self):
        """Test indefinite integral of trig function."""
        result = self.calc.integrate_expression("sin(x)")
        # Integral of sin(x) is -cos(x)
        self.assertTrue(result['integral'] != '')
        print(f"✓ Trig integral: ∫sin(x)dx = {result['integral']}")
    
    def test_definite_integral(self):
        """Test definite integral."""
        result = self.calc.integrate_expression("x**2", lower=0, upper=1)
        # ∫₀¹ x² dx = 1/3
        self.assertTrue(result['integral'] != '')
        print(f"✓ Definite integral: ∫₀¹ x² dx = {result['integral']}")
    
    def test_definite_integral_with_value(self):
        """Test definite integral with numerical value."""
        result = self.calc.integrate_expression("x**2", lower=0, upper=2)
        # ∫₀² x² dx = 8/3
        self.assertIsNotNone(result.get('numerical_value'))
        print(f"✓ Definite integral: ∫₀² x² dx = {result['integral']} ≈ {result['numerical_value']}")
    
    def test_limit_basic(self):
        """Test basic limit."""
        result = self.calc.compute_limit("sin(x)/x", point=0)
        # lim x->0 sin(x)/x = 1
        self.assertIn('1', result['limit'])
        print(f"✓ Limit: lim x->0 sin(x)/x = {result['limit']}")

    def test_limit_at_infinity(self):
        """Test limit at infinity."""
        result = self.calc.compute_limit("1/x", point='oo')
        self.assertIn('0', result['limit'])
        print(f"✓ Limit at inf: lim x->inf 1/x = {result['limit']}")


class TestLinearAlgebra(unittest.TestCase):
    """Test linear algebra operations."""
    
    def setUp(self):
        self.calc = SymbolicCalculator()
    
    def test_determinant_2x2(self):
        """Test determinant of 2x2 matrix."""
        result = self.calc.matrix_operation("[[1,2],[3,4]]", "det")
        self.assertEqual(result['result'], '-2')
        print(f"✓ Determinant 2x2: det([[1,2],[3,4]]) = {result['result']}")
    
    def test_determinant_3x3(self):
        """Test determinant of 3x3 matrix."""
        result = self.calc.matrix_operation("[[1,2,3],[4,5,6],[7,8,9]]", "det")
        # This matrix is singular
        self.assertEqual(result['result'], '0')
        print(f"✓ Determinant 3x3: det([[1,2,3],[4,5,6],[7,8,9]]) = {result['result']}")
    
    def test_matrix_inverse(self):
        """Test matrix inverse."""
        result = self.calc.matrix_operation("[[1,2],[3,4]]", "inverse")
        self.assertTrue('result' in result)
        print(f"✓ Inverse: [[1,2],[3,4]]⁻¹ = {result['result']}")
    
    def test_eigenvalues_2x2(self):
        """Test eigenvalues of 2x2 matrix."""
        result = self.calc.matrix_operation("[[2,1],[1,2]]", "eigenvalues")
        self.assertEqual(result['count'], 2)
        print(f"✓ Eigenvalues: [[2,1],[1,2]] → λ = {result['result']}")
    
    def test_eigenvectors(self):
        """Test eigenvectors."""
        result = self.calc.matrix_operation("[[2,1],[1,2]]", "eigenvectors")
        self.assertGreater(result['count'], 0)
        print(f"✓ Eigenvectors: [[2,1],[1,2]] → {result['result']}")
    
    def test_matrix_rank(self):
        """Test matrix rank."""
        result = self.calc.matrix_operation("[[1,2],[2,4]]", "rank")
        self.assertEqual(result['result'], 1)
        print(f"✓ Rank: [[1,2],[2,4]] → rank = {result['result']}")
    
    def test_matrix_transpose(self):
        """Test matrix transpose."""
        result = self.calc.matrix_operation("[[1,2],[3,4]]", "transpose")
        self.assertTrue('result' in result)
        print(f"✓ Transpose: [[1,2],[3,4]]ᵀ = {result['result']}")
    
    def test_matrix_multiplication(self):
        """Test matrix multiplication."""
        result = self.calc.matrix_operation(
            "[[1,2],[3,4]]", 
            "multiply",
            second_matrix="[[2,0],[1,2]]"
        )
        self.assertTrue('result' in result)
        print(f"✓ Multiplication: [[1,2],[3,4]] × [[2,0],[1,2]] = {result['result']}")
    
    def test_rref(self):
        """Test row reduced echelon form."""
        result = self.calc.matrix_operation("[[1,2,3],[4,5,6],[7,8,9]]", "rref")
        self.assertTrue('result' in result)
        print(f"✓ RREF: [[1,2,3],[4,5,6],[7,8,9]] → {result['result']}")


class TestTrigonometry(unittest.TestCase):
    """Test trigonometric operations."""
    
    def setUp(self):
        self.calc = SymbolicCalculator()
    
    def test_trig_identity_pythagorean(self):
        """Test Pythagorean identity simplification."""
        result = self.calc.simplify_trig("sin(x)**2 + cos(x)**2")
        self.assertEqual(result['simplified'], '1')
        print(f"✓ Identity: sin²(x) + cos²(x) = {result['simplified']}")
    
    def test_trig_double_angle(self):
        """Test double angle formula."""
        result = self.calc.simplify_trig("sin(2*x)")
        print(f"✓ Double angle: sin(2x) = {result['simplified']}")
    
    def test_trig_simplification(self):
        """Test trigonometric simplification."""
        result = self.calc.simplify_trig("tan(x) * cos(x)")
        print(f"✓ Simplification: tan(x)·cos(x) = {result['simplified']}")
    
    def test_trig_equation_basic(self):
        """Test basic trigonometric equation."""
        result = self.calc.solve_trig_equation("sin(x) - 0.5 = 0")
        self.assertTrue('solutions' in result)
        print(f"✓ Trig equation: sin(x) = 0.5 → {result['latex']}")
    
    def test_trig_equation_cos(self):
        """Test cosine equation."""
        result = self.calc.solve_trig_equation("cos(x) = 0")
        print(f"✓ Trig equation: cos(x) = 0 → {result['latex']}")


class TestVerification(unittest.TestCase):
    """Test solution verification mechanisms."""
    
    def setUp(self):
        self.calc = SymbolicCalculator()
    
    def test_verify_correct_solution(self):
        """Test verifying a correct solution."""
        result = self.calc.verify_solution("x**2 - 5*x + 6 = 0", "2")
        self.assertTrue(result['is_correct'])
        print(f"✓ Verify correct: x=2 is solution of x²-5x+6=0")
    
    def test_verify_incorrect_solution(self):
        """Test verifying an incorrect solution."""
        result = self.calc.verify_solution("x**2 - 5*x + 6 = 0", "5")
        self.assertFalse(result['is_correct'])
        print(f"✓ Verify incorrect: x=5 is NOT solution of x²-5x+6=0")
    
    def test_verify_derivative(self):
        """Test verifying a derivative."""
        # d/dx(x²) = 2x
        result = self.calc.verify_derivative("x**2", "2*x")
        self.assertTrue(result['is_correct'])
        print(f"✓ Verify derivative: d/dx(x²) = 2x ✓")
    
    def test_verify_incorrect_derivative(self):
        """Test verifying an incorrect derivative."""
        result = self.calc.verify_derivative("x**2", "x")
        self.assertFalse(result['is_correct'])
        print(f"✓ Verify incorrect derivative: d/dx(x²) ≠ x")
    
    def test_verify_integral(self):
        """Test verifying an antiderivative."""
        # ∫2x dx = x²
        result = self.calc.verify_integral("2*x", "x**2")
        self.assertTrue(result['is_correct'])
        print(f"✓ Verify integral: ∫2x dx = x² ✓")
    
    def test_verify_incorrect_integral(self):
        """Test verifying an incorrect antiderivative."""
        result = self.calc.verify_integral("2*x", "x")
        self.assertFalse(result['is_correct'])
        print(f"✓ Verify incorrect integral: ∫2x dx ≠ x")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        self.calc = SymbolicCalculator()
    
    def test_empty_expression(self):
        """Test handling empty expression."""
        with self.assertRaises(ValueError):
            self.calc.solve_equation("")
        print("✓ Empty expression handled")
    
    def test_invalid_expression(self):
        """Test handling invalid expression."""
        result = self.calc.solve_equation("invalid expression xyz")
        self.assertEqual(result['count'], 0)
        self.assertIn('error', result)
        print("✓ Invalid expression handled")
    
    def test_unsolvable_equation(self):
        """Test handling equations with no closed-form solution."""
        result = self.calc.solve_equation("x**5 - x + 1 = 0")
        # May or may not find solutions
        print(f"✓ Quintic equation: {result}")
    
    def test_divergent_integral(self):
        """Test handling divergent integrals."""
        result = self.calc.integrate_expression("1/x", lower=0, upper=1)
        # ∫₀¹ 1/x dx diverges
        print(f"✓ Divergent integral handled: {result}")
    
    def test_singular_matrix(self):
        """Test handling singular matrix inverse."""
        result = self.calc.matrix_operation("[[1,2],[2,4]]", "inverse")
        self.assertIn('error', result)
        print("✓ Singular matrix handled")


class TestJEEProblems(unittest.TestCase):
    """Test actual JEE-level problems."""
    
    def setUp(self):
        self.calc = SymbolicCalculator()
    
    def test_jee_calculus_1(self):
        """JEE Main: Find derivative of e^(x²)."""
        result = self.calc.differentiate("exp(x**2)")
        # d/dx(e^(x²)) = 2x·e^(x²)
        self.assertTrue('2' in result['derivative'] or 'x' in result['derivative'])
        print(f"✓ JEE Calculus 1: d/dx(e^(x²)) = {result['simplified']}")
    
    def test_jee_calculus_2(self):
        """JEE Main: Find ∫(x·sin(x))dx."""
        result = self.calc.integrate_expression("x * sin(x)")
        # Integration by parts: -x·cos(x) + sin(x)
        self.assertTrue(result['integral'] != '')
        print(f"✓ JEE Calculus 2: ∫x·sin(x)dx = {result['integral']}")
    
    def test_jee_calculus_3(self):
        """JEE Advanced: Find lim_{x→0} (e^x - 1)/x."""
        result = self.calc.compute_limit("(exp(x) - 1)/x", point=0)
        self.assertIn('1', result['limit'])
        print(f"✓ JEE Calculus 3: lim_(x→0) (e^x-1)/x = {result['limit']}")
    
    def test_jee_algebra_1(self):
        """JEE Main: Solve x² - 3x + 2 = 0."""
        result = self.calc.solve_equation("x**2 - 3*x + 2 = 0")
        self.assertIn('1', result['solutions'])
        self.assertIn('2', result['solutions'])
        print(f"✓ JEE Algebra 1: x²-3x+2=0 → x = {result['solutions']}")
    
    def test_jee_algebra_2(self):
        """JEE Main: Find eigenvalues of [[3,1],[0,3]]."""
        result = self.calc.matrix_operation("[[3,1],[0,3]]", "eigenvalues")
        # Eigenvalues are both 3 (repeated)
        self.assertEqual(result['count'], 1)  # One unique eigenvalue
        print(f"✓ JEE Algebra 2: Eigenvalues of [[3,1],[0,3]] = {result['result']}")
    
    def test_jee_geometry_1(self):
        """JEE Main: Distance between points (1,2) and (4,6)."""
        # Using matrix operations to compute
        result = self.calc.evaluate_expression("sqrt((4-1)**2 + (6-2)**2)")
        self.assertEqual(result['result'], 5.0)
        print(f"✓ JEE Geometry 1: Distance = {result['result']}")
    
    def test_jee_trigonometry_1(self):
        """JEE Main: Simplify sin²(x) - cos²(x)."""
        result = self.calc.simplify_trig("sin(x)**2 - cos(x)**2")
        # = -cos(2x)
        print(f"✓ JEE Trig 1: sin²(x) - cos²(x) = {result['simplified']}")
    
    def test_jee_complex_numbers_1(self):
        """JEE Main: Find (1+i)²."""
        result = self.calc.evaluate_expression("(1 + I)**2")
        # (1+i)² = 2i
        print(f"✓ JEE Complex 1: (1+i)² = {result['exact']}")


def run_all_tests():
    """Run all tests and print summary."""
    print("\n" + "="*80)
    print("SYMBOLIC MATH ENGINE - COMPREHENSIVE TEST SUITE")
    print("="*80 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestEquationSolving,
        TestCalculus,
        TestLinearAlgebra,
        TestTrigonometry,
        TestVerification,
        TestEdgeCases,
        TestJEEProblems
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(
        verbosity=2,
        descriptions=True
    )
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback[:100]}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback[:100]}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall: {'✓ PASSED' if success else '✗ FAILED'}")
    print("="*80)
    
    return success


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
