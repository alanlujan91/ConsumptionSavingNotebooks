"""
Test extending grid_l to improve SEGM coverage
"""

import numpy as np
from G2EGMModel import G2EGMModelClass

print("=" * 70)
print("TEST: Extending grid_l for better SEGM coverage")
print("=" * 70)
print()

# Current SEGM (grid_l = grid_m, goes to m_max=10)
print("Testing CURRENT grid setup:")
print("-" * 70)

segm_old = G2EGMModelClass(
    name='segm_old',
    par={'solmethod': 'SEGM', 'sigma': 0.5, 'T': 10, 'do_print': False}
)
segm_old.solve()

t = 5
c_old = segm_old.sol.c[t]
d_old = segm_old.sol.d[t]
c_pure_old = segm_old.sol.c_pure_c[t]

print(f"grid_l: [0, {segm_old.par.grid_m[-1]:.1f}], N={len(segm_old.par.grid_m)}")
print(f"grid_m: [0, {segm_old.par.m_max:.1f}]")
print(f"Subproblem 1 consumption: c ∈ [{c_pure_old.min():.3f}, {c_pure_old.max():.3f}]")
print(f"Final policies at t={t}:")
print(f"  c: mean={c_old.mean():.3f}, max={c_old.max():.3f}")  
print(f"  d: mean={d_old.mean():.3f}, max={d_old.max():.3f}")

# Calculate how much m exceeds bounds
a_max = segm_old.par.grid_a_pd[-1]
c_max_empirical = c_pure_old.max()
l_max_empirical = a_max + c_max_empirical
d_max_empirical = d_old.max()
m_max_empirical = l_max_empirical + d_max_empirical

print(f"\nEmpirical endogenous range:")
print(f"  l_max = a_max + c_max = {a_max:.1f} + {c_max_empirical:.1f} = {l_max_empirical:.1f}")
print(f"  m_max = l_max + d_max = {l_max_empirical:.1f} + {d_max_empirical:.1f} = {m_max_empirical:.1f}")
print(f"  Exceeds target m_max={segm_old.par.m_max:.1f} by {m_max_empirical - segm_old.par.m_max:.1f}")

# Now let's try extending grid_l manually
print("\n" + "=" * 70)
print("PROPOSAL: Reduce grid_l max to leave room for d")
print("=" * 70)
print()

# The idea: if grid_l goes up to (m_max - d_typical), then m = l + d won't exceed m_max
d_typical = 2.0  # Typical max pension contribution
l_max_proposed = segm_old.par.m_max - d_typical

print(f"Proposed: grid_l ∈ [0, {l_max_proposed:.1f}] (was [0, {segm_old.par.m_max:.1f}])")
print(f"Rationale:")
print(f"  • Typical d_max ≈ {d_typical:.1f}")
print(f"  • If l_max = {l_max_proposed:.1f}, then m = l + d ≤ {l_max_proposed:.1f} + {d_typical:.1f} = {segm_old.par.m_max:.1f}")
print(f"  • Ensures endogenous m stays within target grid range")
print()

print("But wait - this creates a different problem!")
print(f"  • grid_l max = {l_max_proposed:.1f}")
print(f"  • But empirically, l can reach {l_max_empirical:.1f}")
print(f"  • We'd be cutting off {l_max_empirical - l_max_proposed:.1f} units of l!")
print()

print("=" * 70)
print("ALTERNATIVE: Extend target m grid for SEGM")
print("=" * 70)
print()

m_max_needed = l_max_empirical + d_max_empirical
m_max_proposed = np.ceil(m_max_needed)

print(f"Empirical need: m can reach {m_max_needed:.1f}")
print(f"Current m_max: {segm_old.par.m_max:.1f}")
print(f"Proposed m_max for SEGM: {m_max_proposed:.1f}")
print()

print("Implications:")
print(f"  • Increases grid size slightly ({m_max_proposed/segm_old.par.m_max:.1%})")
print(f"  • Allows SEGM's endogenous grid to cover full range")
print(f"  • Upper envelope can properly interpolate to target region")
print(f"  • No information loss at high (n, m) states")
print()

# Check solution quality comparison
print("=" * 70)
print("COMPARISON WITH NEGM")
print("=" * 70)
print()

negm = G2EGMModelClass(
    name='negm',
    par={'solmethod': 'NEGM', 'sigma': 0.5, 'T': 10, 'do_print': False}
)
negm.solve()

c_negm = negm.sol.c[t]
d_negm = negm.sol.d[t]

diff_c = np.abs(c_old - c_negm)
diff_d = np.abs(d_old - d_negm)

print(f"At t={t}, current SEGM vs NEGM:")
print(f"  |Δc|: mean={diff_c.mean():.6f}, max={diff_c.max():.6f}")
print(f"  |Δd|: mean={diff_d.mean():.6f}, max={diff_d.max():.6f}")

# Find where differences are largest
worst_idx = np.argmax(diff_d.ravel())
i_n, i_m = np.unravel_index(worst_idx, diff_d.shape)
n_worst = segm_old.par.grid_n[i_n]
m_worst = segm_old.par.grid_m[i_m]

print(f"\nWorst point: (n={n_worst:.2f}, m={m_worst:.2f})")
print(f"  NEGM: c={c_negm[i_n, i_m]:.4f}, d={d_negm[i_n, i_m]:.4f}")
print(f"  SEGM: c={c_old[i_n, i_m]:.4f}, d={d_old[i_n, i_m]:.4f}")
print(f"  Diff: Δc={diff_c[i_n, i_m]:.4f}, Δd={diff_d[i_n, i_m]:.4f}")

# Check if this is in high-m region
if m_worst > 0.8 * segm_old.par.m_max:
    print(f"  ⚠ Problem occurs at high m ({m_worst:.1f} > 80% of m_max)")
    print(f"    This suggests grid coverage issue at boundary")

print()
print("=" * 70)
print("RECOMMENDED ACTION")
print("=" * 70)
print()
print("For SEGM specifically, extend m_max to accommodate endogenous grid:")
print(f"  1. Set m_max_segm = {m_max_proposed:.0f} (vs current {segm_old.par.m_max:.0f})")
print("  2. Keep grid_l = grid_m but with larger range")
print("  3. This allows proper coverage without artificial truncation")
print()
print("Implementation: In G2EGMModel.allocate(), for SEGM:")
print("  - Use larger m_max when creating grids")
print("  - Or extend validation to accept m up to empirical maximum")
print()

