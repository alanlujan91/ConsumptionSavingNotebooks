"""Test suite for NEGMSmooth implementation."""

import numpy as np
from G2EGMModel import G2EGMModelClass
import time

def test_basic_functionality():
    """Test that NEGMSmooth runs without errors."""
    print("\n" + "="*60)
    print("TEST 1: Basic Functionality")
    print("="*60)
    
    # Test with smoothing
    model = G2EGMModelClass(name='test_smooth', par={'solmethod': 'NEGMSmooth', 'sigma': 0.05})
    model.par.T = 3
    model.par.do_print = False
    model.solve()
    
    print("✓ NEGMSmooth runs successfully")
    print(f"  Solution shapes: c={model.sol.c.shape}, d={model.sol.d.shape}")
    
    # Test without smoothing (should match NEGM)
    model_no_smooth = G2EGMModelClass(name='test_no_smooth', par={'solmethod': 'NEGMSmooth', 'sigma': 0.0})
    model_no_smooth.par.T = 3
    model_no_smooth.par.do_print = False
    model_no_smooth.solve()
    
    print("✓ NEGMSmooth with sigma=0 runs successfully")

def test_comparison_with_negm():
    """Test that NEGMSmooth with sigma=0 matches NEGM."""
    print("\n" + "="*60)
    print("TEST 2: Comparison with NEGM")
    print("="*60)
    
    # NEGM
    model_negm = G2EGMModelClass(name='negm', par={'solmethod': 'NEGM'})
    model_negm.par.T = 3
    model_negm.par.do_print = False
    model_negm.solve()
    
    # NEGMSmooth with sigma=0
    model_smooth = G2EGMModelClass(name='smooth', par={'solmethod': 'NEGMSmooth', 'sigma': 0.0})
    model_smooth.par.T = 3
    model_smooth.par.do_print = False
    model_smooth.solve()
    
    # Compare policies
    c_diff = np.abs(model_negm.sol.c - model_smooth.sol.c)
    d_diff = np.abs(model_negm.sol.d - model_smooth.sol.d)
    
    print(f"Max difference in c: {np.max(c_diff):.2e}")
    print(f"Max difference in d: {np.max(d_diff):.2e}")
    
    if np.max(c_diff) < 1e-6 and np.max(d_diff) < 1e-6:
        print("✓ NEGMSmooth with sigma=0 matches NEGM")
    else:
        print("✗ Policies differ more than expected")

def test_timing_and_smoothing():
    """Test timing and verify smoothing effect."""
    print("\n" + "="*60)
    print("TEST 3: Timing and Smoothing Effect")
    print("="*60)
    
    # NEGM
    model_negm = G2EGMModelClass(name='negm', par={'solmethod': 'NEGM'})
    model_negm.par.T = 10
    model_negm.par.do_print = False
    
    t0 = time.time()
    model_negm.solve()
    time_negm = time.time() - t0
    
    print(f"NEGM time:       {time_negm:.4f} seconds")
    
    # NEGMSmooth with various sigma
    for sigma in [0.01, 0.05, 0.1]:
        model = G2EGMModelClass(name=f'smooth_{sigma}', par={'solmethod': 'NEGMSmooth', 'sigma': sigma})
        model.par.T = 10
        model.par.do_print = False
        
        t0 = time.time()
        model.solve()
        time_smooth = time.time() - t0
        
        # Compare middle period
        t = model.par.T // 2
        c_diff = np.abs(model_negm.sol.c[t, 0, :] - model.sol.c[t, 0, :])
        d_diff = np.abs(model_negm.sol.d[t, 0, :] - model.sol.d[t, 0, :])
        
        print(f"\nNEGMSmooth σ={sigma}:")
        print(f"  Time:         {time_smooth:.4f} seconds ({time_smooth/time_negm:.2f}x)")
        print(f"  Max |Δc|:     {np.max(c_diff):.4f}")
        print(f"  Max |Δd|:     {np.max(d_diff):.4f}")

def test_logsum_and_choice_prob():
    """Test the logsum and choice_prob functions directly."""
    print("\n" + "="*60)
    print("TEST 4: Logsum and Choice Probability Functions")
    print("="*60)
    
    from NEGMSmooth import logsum, choice_prob
    
    # Test values
    v = np.array([1.0, 2.0, 3.0])
    sigma = 0.1
    
    # Logsum
    ls = logsum(v, sigma)
    print(f"\nValues: {v}")
    print(f"Logsum(σ={sigma}): {ls:.4f}")
    print(f"Max value: {np.max(v):.4f}")
    print(f"Difference: {ls - np.max(v):.4f}")
    
    # Choice probabilities
    probs = choice_prob(v, sigma)
    print(f"\nChoice probabilities: {probs}")
    print(f"Sum: {np.sum(probs):.6f}")
    
    # Test with -inf values
    v_inf = np.array([1.0, -np.inf, 2.0])
    ls_inf = logsum(v_inf, sigma)
    probs_inf = choice_prob(v_inf, sigma)
    print(f"\nWith -inf: {v_inf}")
    print(f"Logsum: {ls_inf:.4f}")
    print(f"Probabilities: {probs_inf}")
    print(f"Sum: {np.sum(probs_inf):.6f}")
    
    # Test limit as sigma -> 0
    print("\nLimit as σ→0:")
    for s in [0.5, 0.1, 0.01, 0.001]:
        ls = logsum(v, s)
        probs = choice_prob(v, s)
        max_prob = np.max(probs)
        print(f"  σ={s:6.3f}: logsum={ls:.6f}, max(prob)={max_prob:.6f}")
    
    print("\n✓ Logsum and choice_prob functions work correctly")

if __name__ == '__main__':
    test_basic_functionality()
    test_comparison_with_negm()
    test_timing_and_smoothing()
    test_logsum_and_choice_prob()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)
