"""
Quick Test for Symbolic Math Engine
Tests core functionality for JEE mathematics.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.symbolic_math_simple import SymbolicCalculator

def test_all():
    calc = SymbolicCalculator()
    passed = 0
    failed = 0
    
    print("=" * 60)
    print("SYMBOLIC MATH ENGINE - JEE TEST SUITE")
    print("=" * 60)
    
    # Test 1: Linear equation
    print("\n[1] Testing linear equation: 2*x + 3 = 7")
    result = calc.solve_equation("2*x + 3 = 7")
    if '2' in result['solutions']:
        print("    PASS: x = 2")
        passed += 1
    else:
        print(f"    FAIL: {result}")
        failed += 1
    
    # Test 2: Quadratic equation
    print("\n[2] Testing quadratic: x**2 - 5*x + 6 = 0")
    result = calc.solve_equation("x**2 - 5*x + 6 = 0")
    if '2' in result['solutions'] and '3' in result['solutions']:
        print(f"    PASS: x = {result['solutions']}")
        passed += 1
    else:
        print(f"    FAIL: {result}")
        failed += 1
    
    # Test 3: Derivative
    print("\n[3] Testing derivative: d/dx(x**3 + 2*x**2 + x)")
    result = calc.differentiate("x**3 + 2*x**2 + x")
    if result['derivative']:
        print(f"    PASS: {result['simplified']}")
        passed += 1
    else:
        print(f"    FAIL: {result}")
        failed += 1
    
    # Test 4: Derivative with product rule (JEE level)
    print("\n[4] Testing product rule: d/dx(sin(x**2) * exp(x))")
    result = calc.differentiate("sin(x**2) * exp(x)")
    if result['derivative']:
        print(f"    PASS: {result['simplified']}")
        passed += 1
    else:
        print(f"    FAIL: {result}")
        failed += 1
    
    # Test 5: Integral
    print("\n[5] Testing integral: integral(3*x**2 + 2*x + 1)")
    result = calc.integrate_expression("3*x**2 + 2*x + 1")
    if result['integral']:
        print(f"    PASS: {result['integral']}")
        passed += 1
    else:
        print(f"    FAIL: {result}")
        failed += 1
    
    # Test 6: Definite integral
    print("\n[6] Testing definite integral: integral(x**2, 0, 1)")
    result = calc.integrate_expression("x**2", lower=0, upper=1)
    if result['integral']:
        print(f"    PASS: {result['integral']} = {result.get('numerical_value', 'N/A')}")
        passed += 1
    else:
        print(f"    FAIL: {result}")
        failed += 1
    
    # Test 7: Limit
    print("\n[7] Testing limit: lim sin(x)/x as x->0")
    result = calc.compute_limit("sin(x)/x", point=0)
    if '1' in result['limit']:
        print(f"    PASS: {result['limit']}")
        passed += 1
    else:
        print(f"    FAIL: {result}")
        failed += 1
    
    # Test 8: Matrix determinant
    print("\n[8] Testing matrix determinant: det([[1,2],[3,4]])")
    result = calc.matrix_operation("[[1,2],[3,4]]", "det")
    if result['result'] == '-2':
        print(f"    PASS: {result['result']}")
        passed += 1
    else:
        print(f"    FAIL: {result}")
        failed += 1
    
    # Test 9: Matrix eigenvalues
    print("\n[9] Testing eigenvalues: [[2,1],[1,2]]")
    result = calc.matrix_operation("[[2,1],[1,2]]", "eigenvalues")
    if result['count'] > 0:
        print(f"    PASS: {result['result']}")
        passed += 1
    else:
        print(f"    FAIL: {result}")
        failed += 1
    
    # Test 10: Trig identity
    print("\n[10] Testing trig identity: sin(x)**2 + cos(x)**2")
    result = calc.simplify_trig("sin(x)**2 + cos(x)**2")
    if result['simplified'] == '1':
        print(f"    PASS: {result['simplified']}")
        passed += 1
    else:
        print(f"    FAIL: {result}")
        failed += 1
    
    # Test 11: Verify solution
    print("\n[11] Testing verification: x=2 for x**2 - 5*x + 6 = 0")
    result = calc.verify_solution("x**2 - 5*x + 6 = 0", "2")
    if result['is_correct']:
        print(f"    PASS: Solution verified")
        passed += 1
    else:
        print(f"    FAIL: {result}")
        failed += 1
    
    # Test 12: Verify derivative
    print("\n[12] Testing derivative verification: d/dx(x**2) = 2*x")
    result = calc.verify_derivative("x**2", "2*x")
    if result['is_correct']:
        print(f"    PASS: Derivative verified")
        passed += 1
    else:
        print(f"    FAIL: {result}")
        failed += 1
    
    # Test 13: JEE Calculus
    print("\n[13] JEE Calculus: d/dx(e^(x**2))")
    result = calc.differentiate("exp(x**2)")
    if result['derivative']:
        print(f"    PASS: {result['simplified']}")
        passed += 1
    else:
        print(f"    FAIL: {result}")
        failed += 1
    
    # Test 14: JEE Algebra
    print("\n[14] JEE Algebra: Solve x**2 - 3*x + 2 = 0")
    result = calc.solve_equation("x**2 - 3*x + 2 = 0")
    if '1' in result['solutions'] and '2' in result['solutions']:
        print(f"    PASS: x = {result['solutions']}")
        passed += 1
    else:
        print(f"    FAIL: {result}")
        failed += 1
    
    # Test 15: System of equations
    print("\n[15] System: x+y=5, 2*x-y=1")
    result = calc.solve_system(["x + y = 5", "2*x - y = 1"], ["x", "y"])
    if result['count'] > 0:
        print(f"    PASS: {result['solutions']}")
        passed += 1
    else:
        print(f"    FAIL: {result}")
        failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 60)
    
    return failed == 0

if __name__ == '__main__':
    success = test_all()
    sys.exit(0 if success else 1)
