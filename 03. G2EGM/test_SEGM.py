"""
Test suite for Sequential EGM (SEGM)

This test validates that SEGM produces very similar results to NEGM and G2EGM.
SEGM should be:
- Faster than NEGM (no VFI in Step 2)
- Similar accuracy to NEGM
- Similar or slightly faster than G2EGM depending on grid sizes
"""

import numpy as np
import time
import sys
from G2EGMModel import G2EGMModelClass


def compare_solutions(model1, model2, name1, name2, t=0, tol=1e-3):
    """Compare policy functions between two models"""
    print(f"\nComparing {name1} vs {name2} at period t={t}:")
    
    # Get common grid points for comparison
    par = model1.par
    sol1 = model1.sol
    sol2 = model2.sol
    
    # Compare at middle of grid
    i_n = par.Nn // 2
    i_m = par.Nm // 2
    
    n_val = par.grid_n[i_n]
    m_val = par.grid_m[i_m]
    
    # Policy functions
    c1 = sol1.c[t, i_n, i_m]
    c2 = sol2.c[t, i_n, i_m]
    d1 = sol1.d[t, i_n, i_m]
    d2 = sol2.d[t, i_n, i_m]
    # Compute value from inverse value
    v1 = -1.0 / sol1.inv_v[t, i_n, i_m]
    v2 = -1.0 / sol2.inv_v[t, i_n, i_m]
    
    print(f"  At (n={n_val:.3f}, m={m_val:.3f}):")
    print(f"    c: {name1}={c1:.6f}, {name2}={c2:.6f}, diff={abs(c1-c2):.2e}")
    print(f"    d: {name1}={d1:.6f}, {name2}={d2:.6f}, diff={abs(d1-d2):.2e}")
    print(f"    v: {name1}={v1:.6f}, {name2}={v2:.6f}, diff={abs(v1-v2):.2e}")
    
    # Max differences across grid
    c_diff_max = np.nanmax(np.abs(sol1.c[t] - sol2.c[t]))
    d_diff_max = np.nanmax(np.abs(sol1.d[t] - sol2.d[t]))
    # Compute value from inverse value
    v1_grid = -1.0 / sol1.inv_v[t]
    v2_grid = -1.0 / sol2.inv_v[t]
    v_diff_max = np.nanmax(np.abs(v1_grid - v2_grid))
    
    print(f"\n  Max differences across grid:")
    print(f"    |c1 - c2|_max = {c_diff_max:.2e}")
    print(f"    |d1 - d2|_max = {d_diff_max:.2e}")
    print(f"    |v1 - v2|_max = {v_diff_max:.2e}")
    
    # Check if differences are within tolerance
    success = True
    if c_diff_max > tol:
        print(f"  ⚠ WARNING: Consumption difference {c_diff_max:.2e} exceeds tolerance {tol}")
        success = False
    if d_diff_max > tol:
        print(f"  ⚠ WARNING: Pension difference {d_diff_max:.2e} exceeds tolerance {tol}")
        success = False
    if v_diff_max > tol:
        print(f"  ⚠ WARNING: Value difference {v_diff_max:.2e} exceeds tolerance {tol}")
        success = False
    
    if success:
        print(f"  ✓ All differences within tolerance {tol}")
    
    return success


def test_segm_vs_negm():
    """Test that SEGM produces similar results to NEGM"""
    print("="*70)
    print("TEST 1: SEGM vs NEGM")
    print("="*70)
    
    # Configuration for testing
    config = {
        'T': 5,
        'Nm': 200,
        'do_print': False,
        'sigma': 0.0  # No smoothing for cleaner comparison
    }
    
    print("\nSolving with NEGM...")
    model_negm = G2EGMModelClass(name='test_negm')
    for key, val in config.items():
        setattr(model_negm.par, key, val)
    model_negm.par.solmethod = 'NEGM'
    model_negm.allocate()
    
    t0 = time.time()
    model_negm.solve()
    time_negm = time.time() - t0
    print(f"  Time: {time_negm:.3f} seconds")
    
    print("\nSolving with SEGM...")
    model_segm = G2EGMModelClass(name='test_segm')
    for key, val in config.items():
        setattr(model_segm.par, key, val)
    model_segm.par.solmethod = 'SEGM'
    model_segm.allocate()
    
    t0 = time.time()
    model_segm.solve()
    time_segm = time.time() - t0
    print(f"  Time: {time_segm:.3f} seconds ({time_segm/time_negm:.2f}x)")
    
    # Compare solutions at multiple periods
    success = True
    for t in [0, model_negm.par.T // 2]:
        if not compare_solutions(model_negm, model_segm, 'NEGM', 'SEGM', t=t, tol=1e-3):
            success = False
    
    if success:
        print("\n" + "="*70)
        print("✓ SEGM vs NEGM: PASSED")
        print(f"  SEGM speedup: {time_negm/time_segm:.2f}x")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("✗ SEGM vs NEGM: FAILED - differences exceed tolerance")
        print("="*70)
    
    return success


def test_segm_vs_g2egm():
    """Test that SEGM produces similar results to G2EGM"""
    print("\n" + "="*70)
    print("TEST 2: SEGM vs G2EGM")
    print("="*70)
    
    # Configuration for testing
    config = {
        'T': 5,
        'Nm': 200,
        'do_print': False,
        'sigma': 0.0  # No smoothing for cleaner comparison
    }
    
    print("\nSolving with G2EGM...")
    model_g2egm = G2EGMModelClass(name='test_g2egm')
    for key, val in config.items():
        setattr(model_g2egm.par, key, val)
    model_g2egm.par.solmethod = 'G2EGM'
    model_g2egm.allocate()
    
    t0 = time.time()
    model_g2egm.solve()
    time_g2egm = time.time() - t0
    print(f"  Time: {time_g2egm:.3f} seconds")
    
    print("\nSolving with SEGM...")
    model_segm = G2EGMModelClass(name='test_segm')
    for key, val in config.items():
        setattr(model_segm.par, key, val)
    model_segm.par.solmethod = 'SEGM'
    model_segm.allocate()
    
    t0 = time.time()
    model_segm.solve()
    time_segm = time.time() - t0
    print(f"  Time: {time_segm:.3f} seconds ({time_segm/time_g2egm:.2f}x)")
    
    # Compare solutions at multiple periods
    success = True
    for t in [0, model_g2egm.par.T // 2]:
        if not compare_solutions(model_g2egm, model_segm, 'G2EGM', 'SEGM', t=t, tol=1e-3):
            success = False
    
    if success:
        print("\n" + "="*70)
        print("✓ SEGM vs G2EGM: PASSED")
        print(f"  Relative speed: {time_g2egm/time_segm:.2f}x")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("✗ SEGM vs G2EGM: FAILED - differences exceed tolerance")
        print("="*70)
    
    return success


def test_all_three():
    """Compare all three methods together"""
    print("\n" + "="*70)
    print("TEST 3: Three-way comparison (G2EGM vs NEGM vs SEGM)")
    print("="*70)
    
    # Configuration for testing
    config = {
        'T': 5,
        'Nm': 200,
        'do_print': False,
        'sigma': 0.0
    }
    
    models = {}
    times = {}
    
    for method in ['G2EGM', 'NEGM', 'SEGM']:
        print(f"\nSolving with {method}...")
        model = G2EGMModelClass(name=f'test_{method.lower()}')
        for key, val in config.items():
            setattr(model.par, key, val)
        model.par.solmethod = method
        model.allocate()
        
        t0 = time.time()
        model.solve()
        times[method] = time.time() - t0
        models[method] = model
        print(f"  Time: {times[method]:.3f} seconds")
    
    # Print timing summary
    print("\n" + "-"*70)
    print("Timing Summary:")
    print("-"*70)
    baseline = times['NEGM']
    for method in ['G2EGM', 'NEGM', 'SEGM']:
        speedup = baseline / times[method]
        print(f"  {method:8s}: {times[method]:6.3f}s  ({speedup:5.2f}x relative to NEGM)")
    
    # Compare SEGM to both
    t = 0
    print("\n" + "-"*70)
    print(f"Solution Comparison at t={t}:")
    print("-"*70)
    
    success = True
    success &= compare_solutions(models['NEGM'], models['SEGM'], 'NEGM', 'SEGM', t=t, tol=1e-3)
    success &= compare_solutions(models['G2EGM'], models['SEGM'], 'G2EGM', 'SEGM', t=t, tol=1e-3)
    
    if success:
        print("\n" + "="*70)
        print("✓ THREE-WAY COMPARISON: PASSED")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("✗ THREE-WAY COMPARISON: FAILED")
        print("="*70)
    
    return success


if __name__ == '__main__':
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*20 + "SEGM TEST SUITE" + " "*33 + "║")
    print("╚" + "="*68 + "╝")
    print("\n")
    
    all_passed = True
    
    # Run tests
    try:
        all_passed &= test_segm_vs_negm()
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_segm_vs_g2egm()
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_all_three()
    except Exception as e:
        print(f"\n✗ TEST 3 FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # Final summary
    print("\n\n")
    print("╔" + "="*68 + "╗")
    if all_passed:
        print("║" + " "*22 + "ALL TESTS PASSED" + " "*30 + "║")
    else:
        print("║" + " "*22 + "SOME TESTS FAILED" + " "*29 + "║")
    print("╚" + "="*68 + "╝")
    print("\n")
    
    sys.exit(0 if all_passed else 1)
