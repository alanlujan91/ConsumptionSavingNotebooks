"""
Find where SEGM differs most from NEGM to identify bug location
"""

import numpy as np
from G2EGMModel import G2EGMModelClass

print("=" * 70)
print("SEGM BUG LOCALIZATION")
print("=" * 70)
print()

# Solve with high taste shocks
sigma = 0.5

negm = G2EGMModelClass(name='negm', par={'solmethod': 'NEGM', 'sigma': sigma, 'T': 15, 'do_print': False})
negm.solve()

segm = G2EGMModelClass(name='segm', par={'solmethod': 'SEGM', 'sigma': sigma, 'T': 15, 'do_print': False})
segm.solve()

# Analyze at t=7
t = 7

c_negm = negm.sol.c[t]
d_negm = negm.sol.d[t]
c_segm = segm.sol.c[t]
d_segm = segm.sol.d[t]

# Find worst differences
diff_c = np.abs(c_negm - c_segm)
diff_d = np.abs(d_negm - d_segm)

# Get grid
grid_n = negm.par.grid_n
grid_m = negm.par.grid_m

# Find top 10 worst pension differences
worst_d_indices = np.argsort(diff_d.ravel())[-10:][::-1]

print("Top 10 worst pension (d) differences:")
print("-" * 70)
for rank, idx in enumerate(worst_d_indices, 1):
    i_n, i_m = np.unravel_index(idx, diff_d.shape)
    n_val = grid_n[i_n]
    m_val = grid_m[i_m]
    
    print(f"\n#{rank}: State (n={n_val:.3f}, m={m_val:.3f})")
    print(f"  NEGM: c={c_negm[i_n, i_m]:.4f}, d={d_negm[i_n, i_m]:.4f}")
    print(f"  SEGM: c={c_segm[i_n, i_m]:.4f}, d={d_segm[i_n, i_m]:.4f}")
    print(f"  Diff: Δc={diff_c[i_n, i_m]:.4f}, Δd={diff_d[i_n, i_m]:.4f}")

# Check if problems are at boundaries or interior
print("\n" + "=" * 70)
print("SPATIAL DISTRIBUTION OF ERRORS")
print("=" * 70)

# Divide state space into regions
Nn = len(grid_n)
Nm = len(grid_m)

regions = {
    'Low n, Low m': (slice(0, Nn//3), slice(0, Nm//3)),
    'Low n, Mid m': (slice(0, Nn//3), slice(Nm//3, 2*Nm//3)),
    'Low n, High m': (slice(0, Nn//3), slice(2*Nm//3, None)),
    'Mid n, Low m': (slice(Nn//3, 2*Nn//3), slice(0, Nm//3)),
    'Mid n, Mid m': (slice(Nn//3, 2*Nn//3), slice(Nm//3, 2*Nm//3)),
    'Mid n, High m': (slice(Nn//3, 2*Nn//3), slice(2*Nm//3, None)),
    'High n, Low m': (slice(2*Nn//3, None), slice(0, Nm//3)),
    'High n, Mid m': (slice(2*Nn//3, None), slice(Nm//3, 2*Nm//3)),
    'High n, High m': (slice(2*Nn//3, None), slice(2*Nm//3, None)),
}

print("\nMean absolute pension differences by region:")
for name, (n_slice, m_slice) in regions.items():
    region_diff = diff_d[n_slice, m_slice]
    mean_diff = np.mean(region_diff)
    max_diff = np.max(region_diff)
    problem_pct = 100 * (region_diff > 0.1).sum() / region_diff.size
    
    print(f"  {name:20s}: mean={mean_diff:.4f}, max={max_diff:.4f}, {problem_pct:5.1f}% >0.1")

# Check intermediate values at worst point
print("\n" + "=" * 70)
print("DETAILED DIAGNOSIS AT WORST POINT")
print("=" * 70)

worst_idx = np.argmax(diff_d.ravel())
i_n_worst, i_m_worst = np.unravel_index(worst_idx, diff_d.shape)
n_worst = grid_n[i_n_worst]
m_worst = grid_m[i_m_worst]

print(f"\nWorst point: (n={n_worst:.3f}, m={m_worst:.3f})")
print(f"  Grid indices: i_n={i_n_worst}/{Nn}, i_m={i_m_worst}/{Nm}")

print(f"\nPolicies:")
print(f"  NEGM: c={c_negm[i_n_worst, i_m_worst]:.6f}, d={d_negm[i_n_worst, i_m_worst]:.6f}")
print(f"  SEGM: c={c_segm[i_n_worst, i_m_worst]:.6f}, d={d_segm[i_n_worst, i_m_worst]:.6f}")
print(f"  Diff: Δc={diff_c[i_n_worst, i_m_worst]:.6f}, Δd={diff_d[i_n_worst, i_m_worst]:.6f}")

# Check post-decision values at this point
wa_negm = negm.sol.wa[t]
wb_negm = negm.sol.wb[t]
wa_segm = segm.sol.wa[t]
wb_segm = segm.sol.wb[t]

# Compute implied post-decision states for NEGM solution
c_n = c_negm[i_n_worst, i_m_worst]
d_n = d_negm[i_n_worst, i_m_worst]
a_n = m_worst - c_n - d_n

# For SEGM solution
c_s = c_segm[i_n_worst, i_m_worst]
d_s = d_segm[i_n_worst, i_m_worst]
a_s = m_worst - c_s - d_s

print(f"\nImplied post-decision liquid assets:")
print(f"  NEGM: a = m - c - d = {m_worst:.3f} - {c_n:.3f} - {d_n:.3f} = {a_n:.3f}")
print(f"  SEGM: a = m - c - d = {m_worst:.3f} - {c_s:.3f} - {d_s:.3f} = {a_s:.3f}")

# Check if values are valid (not zero or NaN)
print(f"\nSolution quality:")
print(f"  NEGM: c>0? {c_n > 0}, d>=0? {d_n >= 0}, a>=0? {a_n >= 0}")
print(f"  SEGM: c>0? {c_s > 0}, d>=0? {d_s >= 0}, a>=0? {a_s >= 0}")

if c_s == 0 or d_s < 0 or a_s < 0:
    print("\n⚠ SEGM produced infeasible/default solution at this point!")
    print("   This suggests upper envelope failed to cover this region")

# Distribution analysis
print("\n" + "=" * 70)
print("DISTRIBUTION OF DIFFERENCES")
print("=" * 70)

percentiles = [50, 75, 90, 95, 99, 100]
print("\nPension difference (|d_NEGM - d_SEGM|) percentiles:")
for p in percentiles:
    val = np.percentile(diff_d.ravel(), p)
    print(f"  P{p:3d}: {val:.6f}")

# Count problematic points
threshold_values = [0.001, 0.01, 0.1, 0.5]
print("\nPoints exceeding thresholds:")
for thresh in threshold_values:
    count = (diff_d > thresh).sum()
    pct = 100 * count / diff_d.size
    print(f"  |Δd| > {thresh:5.3f}: {count:6d} points ({pct:5.2f}%)")

print()

