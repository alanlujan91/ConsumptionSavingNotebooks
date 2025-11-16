"""Test baseline G2EGM and NEGM with fresh upstream code - proper initialization"""

import numpy as np
from G2EGMModel import G2EGMModelClass

print("="*70)
print("Testing Fresh Upstream Models (proper init)")
print("="*70)

for method in ['G2EGM', 'NEGM']:
    print(f"\n{method}:")
    
    model = G2EGMModelClass(name=f'test_{method}')
    model.par.solmethod = method  # Set BEFORE calling solve
    model.par.do_print = False
    model.allocate()  # Re-allocate after changing solmethod
    
    try:
        model.solve()
        
        # Check first period (t=0)
        c = model.sol.c[0]
        d = model.sol.d[0]
        
        print(f"  ✓ Solved (T={model.par.T})")
        print(f"  c: [{c.min():.3f}, {c.max():.3f}], mean={c.mean():.3f}")
        print(f"  d: [{d.min():.3f}, {d.max():.3f}], mean={d.mean():.3f}")
        print(f"  d>0.01: {np.sum(d > 0.01)} / {d.size} ({100*np.sum(d > 0.01)/d.size:.1f}%)")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*70}")

