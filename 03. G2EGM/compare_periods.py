"""Compare NEGM vs SEGM period by period with IDENTICAL setup"""

import numpy as np
from G2EGMModel import G2EGMModelClass

print("="*70)
print("NEGM vs SEGM: Period-by-Period Comparison")
print("="*70)

# Solve with NEGM
print("\nSolving with NEGM...")
model_negm = G2EGMModelClass(name='negm_test')
model_negm.par.solmethod = 'NEGM'
model_negm.par.do_print = False
model_negm.solve()

# Get parameters from NEGM model to ensure EXACT match
T = model_negm.par.T
grid_n = model_negm.par.grid_n.copy()
grid_m = model_negm.par.grid_m.copy()
grid_a_pd = model_negm.par.grid_a_pd.copy()
grid_b_pd = model_negm.par.grid_b_pd.copy()

print(f"✓ NEGM solved (T={T})")
print(f"  Grid sizes: n={len(grid_n)}, m={len(grid_m)}")
print(f"  Post-decision: a={len(grid_a_pd)}, b={len(grid_b_pd)}")

# Solve with SEGM using EXACT same setup
print("\nSolving with SEGM (identical setup)...")
model_segm = G2EGMModelClass(name='segm_test')
model_segm.par.solmethod = 'SEGM'
model_segm.par.do_print = False

# Force exact same parameters
model_segm.par.T = T
model_segm.par.Ra = model_negm.par.Ra
model_segm.par.Rb = model_negm.par.Rb
model_segm.par.chi = model_negm.par.chi
model_segm.par.beta = model_negm.par.beta
model_segm.par.rho = model_negm.par.rho

# Force exact same grids
model_segm.par.grid_n = grid_n.copy()
model_segm.par.grid_m = grid_m.copy()
model_segm.par.grid_a_pd = grid_a_pd.copy()
model_segm.par.grid_b_pd = grid_b_pd.copy()
model_segm.par.Nn = len(grid_n)
model_segm.par.Nm = len(grid_m)
model_segm.par.Na_pd = len(grid_a_pd)
model_segm.par.Nb_pd = len(grid_b_pd)

model_segm.solve()
print(f"✓ SEGM solved")

# Compare period by period
print(f"\n{'='*70}")
print("PERIOD-BY-PERIOD RESULTS")
print(f"{'='*70}")

test_periods = [0, 1, 2, 3, 4, 5, 10, 15, 19]

for t in test_periods:
    if t >= T:
        continue
    
    print(f"\n{'─'*70}")
    print(f"Period t={t} (periods until terminal: {T-t})")
    print(f"{'─'*70}")
    
    c_negm = model_negm.sol.c[t]
    d_negm = model_negm.sol.d[t]
    c_segm = model_segm.sol.c[t]
    d_segm = model_segm.sol.d[t]
    
    # Statistics
    print(f"\nNEGM:")
    print(f"  c: min={c_negm.min():.4f}, max={c_negm.max():.4f}, mean={c_negm.mean():.4f}")
    print(f"  d: min={d_negm.min():.4f}, max={d_negm.max():.4f}, mean={d_negm.mean():.4f}")
    print(f"  d>0.01: {np.sum(d_negm > 0.01)} / {d_negm.size}")
    
    print(f"\nSEGM:")
    print(f"  c: min={c_segm.min():.4f}, max={c_segm.max():.4f}, mean={c_segm.mean():.4f}")
    print(f"  d: min={d_segm.min():.4f}, max={d_segm.max():.4f}, mean={d_segm.mean():.4f}")
    print(f"  d>0.01: {np.sum(d_segm > 0.01)} / {d_segm.size}")
    
    # Differences
    diff_c = np.abs(c_negm - c_segm)
    diff_d = np.abs(d_negm - d_segm)
    
    print(f"\nDifference:")
    print(f"  |Δc|: max={diff_c.max():.6f}, mean={diff_c.mean():.6f}")
    print(f"  |Δd|: max={diff_d.max():.6f}, mean={diff_d.mean():.6f}")
    
    if diff_c.max() < 1e-3 and diff_d.max() < 1e-3:
        print(f"  ✓ AGREE (max diff < 0.001)")
    else:
        print(f"  ✗ DISAGREE!")
        
        # Show worst point
        if diff_d.max() > diff_c.max():
            i_max = np.unravel_index(diff_d.argmax(), diff_d.shape)
            var = 'd'
        else:
            i_max = np.unravel_index(diff_c.argmax(), diff_c.shape)
            var = 'c'
        
        n_val = grid_n[i_max[0]]
        m_val = grid_m[i_max[1]]
        print(f"  Worst point (n={n_val:.2f}, m={m_val:.2f}) for {var}:")
        print(f"    NEGM: c={c_negm[i_max]:.4f}, d={d_negm[i_max]:.4f}")
        print(f"    SEGM: c={c_segm[i_max]:.4f}, d={d_segm[i_max]:.4f}")

print(f"\n{'='*70}")
print("COMPARISON COMPLETE")
print(f"{'='*70}")

