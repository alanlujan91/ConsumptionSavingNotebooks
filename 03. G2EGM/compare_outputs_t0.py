"""Compare SEGM with G2EGM and NEGM: Final outputs at t=0 on (n, m) grid"""

import numpy as np
from G2EGMModel import G2EGMModelClass

print("="*70)
print("FINAL OUTPUT COMPARISON: SEGM vs G2EGM vs NEGM at t=0")
print("="*70)

# Solve all three models
results = {}

for method in ['G2EGM', 'NEGM', 'SEGM']:
    print(f"\nSolving {method}...")
    model = G2EGMModelClass(name=f'test_{method}', par={'solmethod': method, 'do_print': False})
    model.solve()
    
    t = 0
    
    # Extract outputs on (n, m) grid
    c = model.sol.c[t].copy()
    d = model.sol.d[t].copy()
    inv_v = model.sol.inv_v[t].copy()
    v = -1.0 / inv_v
    
    # Compute marginal values
    # v_m = dv/dm, v_n = dv/dn
    # Using finite differences
    v_m = np.zeros_like(v)
    v_n = np.zeros_like(v)
    
    grid_n = model.par.grid_n
    grid_m = model.par.grid_m
    
    # v_m: derivative w.r.t. m (second index)
    for i_n in range(len(grid_n)):
        for i_m in range(len(grid_m)):
            if i_m < len(grid_m) - 1:
                v_m[i_n, i_m] = (v[i_n, i_m+1] - v[i_n, i_m]) / (grid_m[i_m+1] - grid_m[i_m])
            else:
                v_m[i_n, i_m] = v_m[i_n, i_m-1]
    
    # v_n: derivative w.r.t. n (first index)
    for i_m in range(len(grid_m)):
        for i_n in range(len(grid_n)):
            if i_n < len(grid_n) - 1:
                v_n[i_n, i_m] = (v[i_n+1, i_m] - v[i_n, i_m]) / (grid_n[i_n+1] - grid_n[i_n])
            else:
                v_n[i_n, i_m] = v_n[i_n-1, i_m]
    
    results[method] = {
        'c': c,
        'd': d,
        'v': v,
        'v_m': v_m,
        'v_n': v_n,
        'inv_v': inv_v,
        'grid_n': grid_n.copy(),
        'grid_m': grid_m.copy(),
    }
    
    print(f"  ✓ Solved (T={model.par.T})")
    print(f"  Policy c: [{c.min():.3f}, {c.max():.3f}], mean={c.mean():.3f}")
    print(f"  Policy d: [{d.min():.3f}, {d.max():.3f}], mean={d.mean():.3f}")
    print(f"  d>0.01: {np.sum(d > 0.01)}/{d.size} ({100*np.sum(d > 0.01)/d.size:.1f}%)")
    print(f"  Value v: [{v.min():.3f}, {v.max():.3f}], mean={v.mean():.3f}")
    print(f"  Marginal v_m: [{v_m.min():.3f}, {v_m.max():.3f}]")
    print(f"  Marginal v_n: [{v_n.min():.3f}, {v_n.max():.3f}]")

# Check agreement between G2EGM and NEGM
print(f"\n{'='*70}")
print("G2EGM vs NEGM Agreement")
print(f"{'='*70}")

diff_c = np.abs(results['G2EGM']['c'] - results['NEGM']['c'])
diff_d = np.abs(results['G2EGM']['d'] - results['NEGM']['d'])
diff_v = np.abs(results['G2EGM']['v'] - results['NEGM']['v'])
diff_v_m = np.abs(results['G2EGM']['v_m'] - results['NEGM']['v_m'])
diff_v_n = np.abs(results['G2EGM']['v_n'] - results['NEGM']['v_n'])

print(f"\n|Δc|: max={diff_c.max():.6f}, mean={diff_c.mean():.6f}")
print(f"|Δd|: max={diff_d.max():.6f}, mean={diff_d.mean():.6f}")
print(f"|Δv|: max={diff_v.max():.6f}, mean={diff_v.mean():.6f}")
print(f"|Δv_m|: max={diff_v_m.max():.6f}, mean={diff_v_m.mean():.6f}")
print(f"|Δv_n|: max={diff_v_n.max():.6f}, mean={diff_v_n.mean():.6f}")

if diff_c.max() < 0.01 and diff_d.max() < 0.01:
    print(f"✓ G2EGM and NEGM agree well")
else:
    print(f"⚠ G2EGM and NEGM have some differences")

# Sample points comparison
grid_n = results['G2EGM']['grid_n']
grid_m = results['G2EGM']['grid_m']

print(f"\n{'='*70}")
print("Sample Points on (n, m) Grid")
print(f"{'='*70}")

# Test points: low, medium, high
test_indices = [
    (0, 0, "Low (n=0, m=0)"),
    (len(grid_n)//4, len(grid_m)//4, "Lower-mid"),
    (len(grid_n)//2, len(grid_m)//2, "Middle"),
    (3*len(grid_n)//4, 3*len(grid_m)//4, "Upper-mid"),
    (len(grid_n)-1, len(grid_m)-1, "High (n=max, m=max)"),
]

for i_n, i_m, label in test_indices:
    n_val = grid_n[i_n]
    m_val = grid_m[i_m]
    
    print(f"\n{label}: (n={n_val:.2f}, m={m_val:.2f})")
    print(f"{'Method':<10} {'c':<8} {'d':<8} {'v':<10} {'v_m':<8} {'v_n':<8}")
    print("-" * 60)
    
    for method in ['G2EGM', 'NEGM']:
        c = results[method]['c'][i_n, i_m]
        d = results[method]['d'][i_n, i_m]
        v = results[method]['v'][i_n, i_m]
        v_m = results[method]['v_m'][i_n, i_m]
        v_n = results[method]['v_n'][i_n, i_m]
        
        print(f"{method:<10} {c:<8.3f} {d:<8.3f} {v:<10.3f} {v_m:<8.3f} {v_n:<8.3f}")

print(f"\n{'='*70}")
print("Summary Statistics")
print(f"{'='*70}")

for method in ['G2EGM', 'NEGM']:
    print(f"\n{method}:")
    
    c = results[method]['c']
    d = results[method]['d']
    v = results[method]['v']
    v_m = results[method]['v_m']
    v_n = results[method]['v_n']
    
    print(f"  c:   mean={c.mean():.3f}, std={c.std():.3f}, [{c.min():.3f}, {c.max():.3f}]")
    print(f"  d:   mean={d.mean():.3f}, std={d.std():.3f}, [{d.min():.3f}, {d.max():.3f}]")
    print(f"  v:   mean={v.mean():.3f}, std={v.std():.3f}, [{v.min():.3f}, {v.max():.3f}]")
    print(f"  v_m: mean={v_m.mean():.3f}, std={v_m.std():.3f}, [{v_m.min():.3f}, {v_m.max():.3f}]")
    print(f"  v_n: mean={v_n.mean():.3f}, std={v_n.std():.3f}, [{v_n.min():.3f}, {v_n.max():.3f}]")
    print(f"  d>0.01: {np.sum(d > 0.01)}/{d.size} ({100*np.sum(d > 0.01)/d.size:.1f}%)")

print(f"\n{'='*70}")

