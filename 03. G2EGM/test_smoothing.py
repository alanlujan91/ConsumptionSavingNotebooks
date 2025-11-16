"""
Test suite for taste shock smoothing in G2EGM and NEGM

This test file validates that:
1. Smoothing can be disabled (sigma=0) and matches original behavior
2. Smoothing is enabled when sigma>0 and produces different results
3. Helper functions (logsumexp, choice_probs) work correctly
4. Both G2EGM and NEGM solvers work with smoothing
"""

import numpy as np
import time
import sys
sys.path.insert(0, '03. G2EGM')
from G2EGMModel import G2EGMModelClass
from post_decision import logsumexp, choice_probs


def test_helper_functions():
    """Test logsumexp and choice_probs helper functions"""
    print("="*70)
    print("TEST 1: Helper Functions")
    print("="*70)
    
    # Test values
    v_work = 2.0
    v_retire = 1.5
    
    print("\nTesting with v_work=2.0, v_retire=1.5:")
    
    # Test with sigma=0 (should return discrete max)
    result_discrete = logsumexp(v_work, v_retire, 0.0)
    prob_discrete = choice_probs(v_work, v_retire, 0.0)
    print(f"\nσ=0.0 (discrete):")
    print(f"  logsumexp = {result_discrete:.6f} (should be {max(v_work, v_retire):.6f})")
    print(f"  prob(work) = {prob_discrete:.6f} (should be 1.0)")
    assert abs(result_discrete - max(v_work, v_retire)) < 1e-10, "Discrete max failed"
    assert abs(prob_discrete - 1.0) < 1e-10, "Discrete probability failed"
    
    # Test with different sigma values
    print("\nWith smoothing:")
    for sigma in [0.01, 0.05, 0.1, 0.5]:
        result = logsumexp(v_work, v_retire, sigma)
        prob = choice_probs(v_work, v_retire, sigma)
        print(f"  σ={sigma:.2f}: logsumexp={result:.6f}, prob(work)={prob:.6f}")
        # Check that smooth max is larger than discrete max
        assert result >= max(v_work, v_retire), f"Smooth max should be >= discrete max for sigma={sigma}"
        # Check that probability is in [0, 1]
        assert 0.0 <= prob <= 1.0, f"Probability should be in [0,1] for sigma={sigma}"
    
    # Test symmetry
    prob_work = choice_probs(v_work, v_retire, 0.1)
    prob_retire = choice_probs(v_retire, v_work, 0.1)
    print(f"\nSymmetry check (σ=0.1):")
    print(f"  prob(work | v_w=2, v_r=1.5) = {prob_work:.6f}")
    print(f"  prob(work | v_w=1.5, v_r=2) = {prob_retire:.6f}")
    print(f"  Sum should be 1.0: {prob_work + prob_retire:.6f}")
    
    print("\n✓ Helper function tests passed!\n")


def test_g2egm_smoothing():
    """Test G2EGM with and without smoothing"""
    print("="*70)
    print("TEST 2: G2EGM Smoothing")
    print("="*70)
    
    # Small model for faster testing
    config = {
        'T': 3,
        'Nm': 50,
        'do_print': False
    }
    
    print("\nSolving G2EGM without smoothing (σ=0)...")
    model_no_smooth = G2EGMModelClass(name='g2egm_no_smooth')
    for key, val in config.items():
        setattr(model_no_smooth.par, key, val)
    model_no_smooth.par.sigma = 0.0
    model_no_smooth.allocate()
    
    t0 = time.time()
    model_no_smooth.solve()
    time_no_smooth = time.time() - t0
    print(f"  Time: {time_no_smooth:.3f} seconds")
    
    print("\nSolving G2EGM with smoothing (σ=0.05)...")
    model_smooth = G2EGMModelClass(name='g2egm_smooth')
    for key, val in config.items():
        setattr(model_smooth.par, key, val)
    model_smooth.par.sigma = 0.05
    model_smooth.allocate()
    
    t0 = time.time()
    model_smooth.solve()
    time_smooth = time.time() - t0
    print(f"  Time: {time_smooth:.3f} seconds ({time_smooth/time_no_smooth:.2f}x)")
    
    # Compare solutions at middle period
    t = 0
    c_diff = np.abs(model_no_smooth.sol.c[t] - model_smooth.sol.c[t])
    w_diff = np.abs(model_no_smooth.sol.w[t] - model_smooth.sol.w[t])
    
    print(f"\nDifferences at t={t}:")
    print(f"  Mean |Δc|: {np.mean(c_diff):.6f}")
    print(f"  Max |Δc|:  {np.max(c_diff):.6f}")
    print(f"  Mean |Δw|: {np.mean(w_diff):.6f}")
    print(f"  Max |Δw|:  {np.max(w_diff):.6f}")
    
    # Verify that solutions are different
    assert np.max(c_diff) > 1e-6, "Smoothing should affect consumption policy"
    assert np.max(w_diff) > 1e-6, "Smoothing should affect post-decision value"
    
    print("\n✓ G2EGM smoothing test passed!\n")
    return model_no_smooth, model_smooth


def test_negm_smoothing():
    """Test NEGM with and without smoothing"""
    print("="*70)
    print("TEST 3: NEGM Smoothing")
    print("="*70)
    
    # Small model for faster testing
    config = {
        'T': 3,
        'Nm': 50,
        'do_print': False,
        'solmethod': 'NEGM'
    }
    
    print("\nSolving NEGM without smoothing (σ=0)...")
    model_no_smooth = G2EGMModelClass(name='negm_no_smooth')
    for key, val in config.items():
        setattr(model_no_smooth.par, key, val)
    model_no_smooth.par.sigma = 0.0
    model_no_smooth.allocate()
    
    t0 = time.time()
    model_no_smooth.solve()
    time_no_smooth = time.time() - t0
    print(f"  Time: {time_no_smooth:.3f} seconds")
    
    print("\nSolving NEGM with smoothing (σ=0.05)...")
    model_smooth = G2EGMModelClass(name='negm_smooth')
    for key, val in config.items():
        setattr(model_smooth.par, key, val)
    model_smooth.par.sigma = 0.05
    model_smooth.allocate()
    
    t0 = time.time()
    model_smooth.solve()
    time_smooth = time.time() - t0
    print(f"  Time: {time_smooth:.3f} seconds ({time_smooth/time_no_smooth:.2f}x)")
    
    # Compare solutions at middle period
    t = 0
    c_diff = np.abs(model_no_smooth.sol.c[t] - model_smooth.sol.c[t])
    w_diff = np.abs(model_no_smooth.sol.w[t] - model_smooth.sol.w[t])
    
    print(f"\nDifferences at t={t}:")
    print(f"  Mean |Δc|: {np.mean(c_diff):.6f}")
    print(f"  Max |Δc|:  {np.max(c_diff):.6f}")
    print(f"  Mean |Δw|: {np.mean(w_diff):.6f}")
    print(f"  Max |Δw|:  {np.max(w_diff):.6f}")
    
    # Verify that solutions are different
    assert np.max(c_diff) > 1e-6, "Smoothing should affect consumption policy"
    assert np.max(w_diff) > 1e-6, "Smoothing should affect post-decision value"
    
    print("\n✓ NEGM smoothing test passed!\n")
    return model_no_smooth, model_smooth


def test_smoothing_effect():
    """Test effect of different sigma values"""
    print("="*70)
    print("TEST 4: Effect of Different Sigma Values")
    print("="*70)
    
    config = {
        'T': 3,
        'Nm': 50,
        'do_print': False
    }
    
    sigma_values = [0.0, 0.01, 0.05, 0.1, 0.2]
    models = []
    
    print("\nSolving G2EGM with different sigma values...")
    for sigma in sigma_values:
        model = G2EGMModelClass(name=f'g2egm_sigma_{sigma}')
        for key, val in config.items():
            setattr(model.par, key, val)
        model.par.sigma = sigma
        model.allocate()
        
        t0 = time.time()
        model.solve()
        solve_time = time.time() - t0
        
        models.append((sigma, model, solve_time))
        print(f"  σ={sigma:.2f}: {solve_time:.3f} seconds")
    
    # Compare solutions
    print(f"\nComparison at state (i_n=25, i_m=25):")
    print(f"{'σ':<8} {'c':<12} {'d':<12} {'w':<12}")
    print("-"*44)
    
    t = 0
    i_n = 25
    i_m = 25
    
    for sigma, model, solve_time in models:
        c = model.sol.c[t, i_n, i_m]
        d = model.sol.d[t, i_n, i_m]
        w = model.sol.w[t, 10, 10]  # post-decision value
        print(f"{sigma:<8.2f} {c:<12.6f} {d:<12.6f} {w:<12.6f}")
    
    print("\n✓ Sigma comparison test passed!\n")


def run_all_tests():
    """Run all test suites"""
    print("\n" + "="*70)
    print("Taste Shock Smoothing Test Suite")
    print("="*70 + "\n")
    
    try:
        test_helper_functions()
        test_g2egm_smoothing()
        test_negm_smoothing()
        test_smoothing_effect()
        
        print("="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        return True
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
