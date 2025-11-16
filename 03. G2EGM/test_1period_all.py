"""Multi-Period Test: Compare NEGM, G2EGM, and SEGM across periods"""

import numpy as np
from G2EGMModel import G2EGMModelClass

print("="*70)
print("MULTI-PERIOD COMPARISON: NEGM vs G2EGM vs SEGM")
print("="*70)

# Common parameters
T = 20  # Full model
methods = ['NEGM', 'SEGM']  # Test NEGM vs SEGM first
all_results = {}

for method in methods:
    print(f"\n{'='*70}")
    print(f"Solving with {method}")
    print(f"{'='*70}")
    
    model = G2EGMModelClass(name=f'test_{method}')
    model.par.T = T
    model.par.solmethod = method
    model.par.do_print = False
    
    # Use default grids (don't override)
    
    try:
        model.solve()
        
        # Store ALL periods
        all_results[method] = {
            'c': [model.sol.c[t].copy() for t in range(T)],
            'd': [model.sol.d[t].copy() for t in range(T)],
            'model': model,
        }
        
        print(f"✓ {method} solved successfully")
        
    except Exception as e:
        print(f"\n✗ {method} failed: {e}")
        import traceback
        traceback.print_exc()
        all_results[method] = None

# Compare results period by period
print(f"\n{'='*70}")
print("PERIOD-BY-PERIOD COMPARISON")
print(f"{'='*70}")

# Check which methods succeeded
success = {m: all_results[m] is not None for m in methods}

if not any(success.values()):
    print("\n✗ All methods failed!")
else:
    # Compare across key periods
    test_periods = [0, 1, 2, 5, 10, 15, 19]
    
    for t in test_periods:
        if t >= T:
            continue
            
        print(f"\n{'─'*70}")
        print(f"Period t = {t} (T-t = {T-t})")
        print(f"{'─'*70}")
        
        # Print statistics for each method
        for method in methods:
            if not success[method]:
                continue
                
            d = all_results[method]['d'][t]
            c = all_results[method]['c'][t]
            
            print(f"\n{method}:")
            print(f"  c: [{c.min():.3f}, {c.max():.3f}], mean={c.mean():.3f}")
            print(f"  d: [{d.min():.3f}, {d.max():.3f}], mean={d.mean():.3f}")
            print(f"  d nonzero (>0.01): {np.sum(d > 0.01)}/{d.size}")
        
        # Compare NEGM vs SEGM if both exist
        if success['NEGM'] and success['SEGM']:
            d_negm = all_results['NEGM']['d'][t]
            d_segm = all_results['SEGM']['d'][t]
            c_negm = all_results['NEGM']['c'][t]
            c_segm = all_results['SEGM']['c'][t]
            
            diff_d = np.abs(d_negm - d_segm)
            diff_c = np.abs(c_negm - c_segm)
            
            print(f"\nNEGM vs SEGM:")
            print(f"  |Δd|: max={diff_d.max():.6f}, mean={diff_d.mean():.6f}")
            print(f"  |Δc|: max={diff_c.max():.6f}, mean={diff_c.mean():.6f}")
            
            if diff_d.max() < 0.01 and diff_c.max() < 0.01:
                print(f"  ✓ Agreement (max diff < 0.01)")
            else:
                print(f"  ✗ DISAGREE!")
                
                # Find point of maximum difference
                i_max = np.unravel_index(diff_d.argmax(), diff_d.shape)
                n_max = all_results['NEGM']['model'].par.grid_n[i_max[0]]
                m_max = all_results['NEGM']['model'].par.grid_m[i_max[1]]
                print(f"    Max diff at (n={n_max:.2f}, m={m_max:.2f}):")
                print(f"      NEGM: c={c_negm[i_max]:.3f}, d={d_negm[i_max]:.3f}")
                print(f"      SEGM: c={c_segm[i_max]:.3f}, d={d_segm[i_max]:.3f}")

print(f"\n{'='*70}")
print("TEST COMPLETED")
print(f"{'='*70}")

