"""Test manual SEGM vs G2EGM and NEGM at t=0"""

import numpy as np
from G2EGMModel import G2EGMModelClass
import manual_segm

print("="*70)
print("MANUAL SEGM vs G2EGM vs NEGM at t=0")
print("="*70)

# Solve baseline models
results = {}

for method in ['G2EGM', 'NEGM']:
    print(f"\nSolving {method}...")
    model = G2EGMModelClass(name=f'test_{method}')
    model.par.solmethod = method
    model.par.do_print = False
    model.allocate()
    model.solve()
    
    t = 0
    results[method] = {
        'c': model.sol.c[t].copy(),
        'd': model.sol.d[t].copy(),
        'v': -1.0 / model.sol.inv_v[t].copy(),
        'w': model.sol.w[t].copy(),
        'wa': model.sol.wa[t].copy(),
        'wb': model.sol.wb[t].copy() if hasattr(model.sol, 'wb') and len(model.sol.wb) > 0 else None,
        'grid_n': model.par.grid_n.copy(),
        'grid_m': model.par.grid_m.copy(),
        'par': model.par,
    }
    
    d = results[method]['d']
    print(f"  ✓ {method} solved")
    print(f"  d: mean={d.mean():.3f}, max={d.max():.3f}")
    print(f"  d>0.01: {np.sum(d > 0.01)}/{d.size} ({100*np.sum(d > 0.01)/d.size:.1f}%)")

# Use G2EGM's post-decision functions for manual SEGM
# Need to create a fresh model for jit access
print(f"\nCreating model for manual SEGM...")
model_for_segm = G2EGMModelClass(name='for_segm')
model_for_segm.par.solmethod = 'G2EGM'
model_for_segm.par.do_print = False
model_for_segm.allocate()

par = results['G2EGM']['par']
w = results['G2EGM']['w']
wa = results['G2EGM']['wa']
wb = results['G2EGM']['wb']

# Run manual SEGM
c_segm, d_segm, v_segm = manual_segm.solve_segm_manual(w, wa, wb, par, model_for_segm)

results['SEGM'] = {
    'c': c_segm,
    'd': d_segm,
    'v': v_segm,
}

# COMPARISON
print(f"\n{'='*70}")
print("COMPARISON")
print(f"{'='*70}")

grid_n = results['G2EGM']['grid_n']
grid_m = results['G2EGM']['grid_m']

for method in ['G2EGM', 'NEGM']:
    c_ref = results[method]['c']
    d_ref = results[method]['d']
    v_ref = results[method]['v']
    
    c_segm = results['SEGM']['c']
    d_segm = results['SEGM']['d']
    v_segm = results['SEGM']['v']
    
    diff_c = np.abs(c_ref - c_segm)
    diff_d = np.abs(d_ref - d_segm)
    diff_v = np.abs(v_ref - v_segm)
    
    print(f"\n{method} vs Manual SEGM:")
    print(f"  |Δc|: max={diff_c.max():.6f}, mean={diff_c.mean():.6f}")
    print(f"  |Δd|: max={diff_d.max():.6f}, mean={diff_d.mean():.6f}")
    print(f"  |Δv|: max={diff_v.max():.6f}, mean={diff_v.mean():.6f}")
    
    if diff_c.max() < 0.1 and diff_d.max() < 0.1:
        print(f"  ✓ GOOD AGREEMENT!")
    elif diff_c.max() < 1.0 and diff_d.max() < 1.0:
        print(f"  ⚠ Moderate agreement")
    else:
        print(f"  ✗ POOR AGREEMENT")
    
    # Show worst points
    i_max_d = np.unravel_index(diff_d.argmax(), diff_d.shape)
    i_max_c = np.unravel_index(diff_c.argmax(), diff_c.shape)
    
    print(f"  Max |Δd| at (n={grid_n[i_max_d[0]]:.2f}, m={grid_m[i_max_d[1]]:.2f}):")
    print(f"    {method}: c={c_ref[i_max_d]:.4f}, d={d_ref[i_max_d]:.4f}, v={v_ref[i_max_d]:.4f}")
    print(f"    SEGM:  c={c_segm[i_max_d]:.4f}, d={d_segm[i_max_d]:.4f}, v={v_segm[i_max_d]:.4f}")

# Sample points
print(f"\n{'='*70}")
print("Sample Points")
print(f"{'='*70}")

test_indices = [
    (len(grid_n)//4, len(grid_m)//4, "Lower-mid"),
    (len(grid_n)//2, len(grid_m)//2, "Middle"),
    (3*len(grid_n)//4, 3*len(grid_m)//4, "Upper-mid"),
]

for i_n, i_m, label in test_indices:
    n_val = grid_n[i_n]
    m_val = grid_m[i_m]
    
    print(f"\n{label}: (n={n_val:.2f}, m={m_val:.2f})")
    print(f"{'Method':<10} {'c':<8} {'d':<8} {'v':<10}")
    print("-" * 40)
    
    for method in ['G2EGM', 'NEGM', 'SEGM']:
        c = results[method]['c'][i_n, i_m]
        d = results[method]['d'][i_n, i_m]
        v = results[method]['v'][i_n, i_m]
        
        print(f"{method:<10} {c:<8.3f} {d:<8.3f} {v:<10.3f}")

print(f"\n{'='*70}")

