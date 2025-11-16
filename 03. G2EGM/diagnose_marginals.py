"""Diagnostic: Check SEGM marginal values"""

import numpy as np
from G2EGMModel import G2EGMModelClass

# Create and solve
model = G2EGMModelClass(name='G2EGM')
model.par.solmethod = 'SEGM'
print(f"Solving with SEGM (χ = {model.par.chi})...")
model.solve()

print("\n" + "="*70)
print("DIAGNOSTIC: Checking Marginal Values")
print("="*70)

# Check a middle period
t = 15
sol = model.sol
par = model.par

print(f"\nPeriod t={t}")
print(f"χ = {par.chi:.2f}")
print(f"For d > 0, need: 1 < v_l/v_b < {par.chi + 1:.2f}")

# Check subproblem 1 outputs (consumption on b, l grid)
try:
    c_pure_c = sol.c_pure_c[t]
    v_l = sol.v_pure_c_m[t]  # marginal value w.r.t. liquid resources
    v_b = sol.v_pure_c_b[t]  # marginal value w.r.t. pension wealth
    
    print(f"\nSubproblem 1: Pure Consumption on (b, l) grid")
    print(f"  c_pure_c shape: {c_pure_c.shape}")
    print(f"  v_l shape: {v_l.shape}")
    print(f"  v_b shape: {v_b.shape}")
    
    # Check for NaN/inf
    print(f"\n  c_pure_c: min={np.nanmin(c_pure_c):.3f}, max={np.nanmax(c_pure_c):.3f}, NaN={np.sum(np.isnan(c_pure_c))}")
    print(f"  v_l: min={np.nanmin(v_l):.6f}, max={np.nanmax(v_l):.6f}, NaN={np.sum(np.isnan(v_l))}")
    print(f"  v_b: min={np.nanmin(v_b):.6f}, max={np.nanmax(v_b):.6f}, NaN={np.sum(np.isnan(v_b))}")
    
    # Compute ratio where both are valid
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = v_l / v_b
    
    valid_mask = np.isfinite(ratio) & (v_l > 1e-10) & (v_b > 1e-10)
    valid_ratios = ratio[valid_mask]
    
    if len(valid_ratios) > 0:
        print(f"\n  Marginal value ratios v_l/v_b (valid points only):")
        print(f"    min={valid_ratios.min():.3f}, max={valid_ratios.max():.3f}, mean={valid_ratios.mean():.3f}")
        
        # Check how many satisfy conditions for d > 0
        in_range = (valid_ratios > 1.0) & (valid_ratios < par.chi + 1)
        print(f"    Points with 1 < ratio < {par.chi+1:.2f}: {np.sum(in_range)}/{len(valid_ratios)}")
        print(f"    Points with ratio <= 1: {np.sum(valid_ratios <= 1.0)}")
        print(f"    Points with ratio >= {par.chi+1:.2f}: {np.sum(valid_ratios >= par.chi + 1)}")
        
        # Sample some specific points
        print(f"\n  Sample points:")
        indices = [(0, 0), (par.Nb_pd//2, par.Nm//2), (par.Nb_pd-1, par.Nm-1)]
        for i_b, i_l in indices:
            if i_b < par.Nb_pd and i_l < par.Nm:
                b = par.grid_b_pd[i_b]
                l = par.grid_m[i_l]
                c_val = c_pure_c[i_b, i_l]
                v_l_val = v_l[i_b, i_l]
                v_b_val = v_b[i_b, i_l]
                
                if v_b_val > 1e-10:
                    r = v_l_val / v_b_val
                    # Compute implied d
                    if v_l_val > v_b_val:
                        d_implied = (par.chi * v_b_val) / (v_l_val - v_b_val) - 1.0
                        d_implied = max(0.0, d_implied)
                    else:
                        d_implied = 0.0
                    
                    print(f"    (b={b:.2f}, l={l:.2f}): c={c_val:.3f}, v_l={v_l_val:.6f}, v_b={v_b_val:.6f}")
                    print(f"      ratio={r:.3f}, d_implied={d_implied:.3f}")
    
    # Check final output
    d_final = sol.d[t]
    print(f"\n  Final d(n, m) output:")
    print(f"    min={d_final.min():.6f}, max={d_final.max():.6f}, mean={d_final.mean():.6f}")
    print(f"    Nonzero (>1e-6): {np.sum(d_final > 1e-6)}/{d_final.size}")
    
    # Compare with NEGM
    print(f"\n" + "="*70)
    print("Comparing with NEGM")
    print("="*70)
    
    model_negm = G2EGMModelClass(name='NEGM')
    model_negm.par.solmethod = 'NEGM'
    print(f"Solving with NEGM...")
    model_negm.solve()
    
    d_negm = model_negm.sol.d[t]
    print(f"\n  NEGM d(n, m) output:")
    print(f"    min={d_negm.min():.6f}, max={d_negm.max():.6f}, mean={d_negm.mean():.6f}")
    print(f"    Nonzero (>1e-6): {np.sum(d_negm > 1e-6)}/{d_negm.size}")
    
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

