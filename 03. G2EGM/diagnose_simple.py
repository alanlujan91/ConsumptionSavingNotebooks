"""Simple diagnostic: Just check if arrays are populated"""

import numpy as np
from G2EGMModel import G2EGMModelClass
import sys

print("Creating model...")
model = G2EGMModelClass(name='G2EGM')
model.par.solmethod = 'SEGM'
model.par.do_print = False

print(f"χ = {model.par.chi:.2f}")
print("Solving...")

try:
    model.solve()
    print("✓ Solve completed without crash")
    
    # Just print shapes and basic stats - don't index into arrays
    t = 10
    print(f"\nChecking period t={t}:")
    
    # Check if arrays exist
    if hasattr(model.sol, 'v_pure_c_m') and len(model.sol.v_pure_c_m) > 0:
        v_l = model.sol.v_pure_c_m[t]
        v_b = model.sol.v_pure_c_b[t]
        
        print(f"  v_l (marginal value liquid): shape={v_l.shape}, dtype={v_l.dtype}")
        print(f"    Contains NaN: {np.any(np.isnan(v_l))}")
        print(f"    Contains Inf: {np.any(np.isinf(v_l))}")
        print(f"    All zeros: {np.all(v_l == 0)}")
        
        print(f"  v_b (marginal value pension): shape={v_b.shape}, dtype={v_b.dtype}")
        print(f"    Contains NaN: {np.any(np.isnan(v_b))}")
        print(f"    Contains Inf: {np.any(np.isinf(v_b))}")
        print(f"    All zeros: {np.all(v_b == 0)}")
        
        # Try to compute ratio safely
        with np.errstate(all='ignore'):
            ratio = v_l / v_b
            valid = np.isfinite(ratio)
            if np.any(valid):
                valid_ratios = ratio[valid]
                print(f"  Ratio v_l/v_b:")
                print(f"    Valid points: {np.sum(valid)}/{ratio.size}")
                print(f"    Range: [{np.min(valid_ratios):.3f}, {np.max(valid_ratios):.3f}]")
                print(f"    Need: (1, {model.par.chi+1:.2f}) for d>0")
            else:
                print(f"  Ratio v_l/v_b: No valid points!")
    else:
        print("  ✗ v_pure_c_m not found or empty!")
    
    d = model.sol.d[t]
    print(f"\n  Final d: shape={d.shape}, dtype={d.dtype}")
    print(f"    Contains NaN: {np.any(np.isnan(d))}")
    print(f"    Max value: {np.nanmax(d):.6f}")
    print(f"    Nonzero: {np.sum(d > 1e-6)}/{d.size}")
    
except Exception as e:
    print(f"\n✗ Error during solve: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ Diagnostic completed")

