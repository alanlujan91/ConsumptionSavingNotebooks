"""
Analyze grid coverage for SEGM's two-stage process
"""

import numpy as np
from G2EGMModel import G2EGMModelClass

print("=" * 70)
print("SEGM GRID COVERAGE ANALYSIS")
print("=" * 70)
print()

# Setup model
segm = G2EGMModelClass(name='segm', par={'solmethod': 'SEGM', 'sigma': 0.5, 'T': 10, 'do_print': False})
segm.solve()

par = segm.par

print("Grid ranges:")
print(f"  Post-decision a: [{par.grid_a_pd[0]:.2f}, {par.grid_a_pd[-1]:.2f}] (N={par.Na_pd})")
print(f"  Post-decision b: [{par.grid_b_pd[0]:.2f}, {par.grid_b_pd[-1]:.2f}] (N={par.Nb_pd})")
print(f"  Intermediate l:  [{par.grid_m[0]:.2f}, {par.grid_m[-1]:.2f}] (N={len(par.grid_m)})")
print(f"  Target n:        [{par.grid_n[0]:.2f}, {par.grid_n[-1]:.2f}] (N={par.Nn})")
print(f"  Target m:        [{par.grid_m[0]:.2f}, {par.grid_m[-1]:.2f}] (N={par.Nm})")
print()

# Theoretical coverage
print("Theoretical endogenous ranges:")
print("-" * 70)

# From (b, a) → (b, l) via l = a + c
# Assume c ∈ [0, 3] (rough estimate)
a_min, a_max = par.grid_a_pd[0], par.grid_a_pd[-1]
c_min, c_max = 0.0, 3.0

l_min_theory = a_min + c_min
l_max_theory = a_max + c_max

print(f"\nStep 1: (b,a) → (b,l) via l = a + c")
print(f"  a ∈ [{a_min:.2f}, {a_max:.2f}]")
print(f"  c ∈ [{c_min:.2f}, {c_max:.2f}] (estimated)")
print(f"  → l ∈ [{l_min_theory:.2f}, {l_max_theory:.2f}] (theoretical)")
print(f"  BUT grid_l ∈ [{par.grid_m[0]:.2f}, {par.grid_m[-1]:.2f}]")

if l_max_theory > par.grid_m[-1]:
    print(f"  ⚠ Mismatch! l can reach {l_max_theory:.2f} but grid only goes to {par.grid_m[-1]:.2f}")

# From (b, l) → (n, m) via m = l + d, n = b - d - ψ(d)
# Assume d ∈ [0, 3] (rough estimate)
b_min, b_max = par.grid_b_pd[0], par.grid_b_pd[-1]
d_min, d_max = 0.0, 3.0
psi_max = par.chi * np.log(1 + d_max)

m_min_theory = l_min_theory + d_min  # = a_min
m_max_theory = l_max_theory + d_max  # = a_max + c_max + d_max

n_min_theory = b_min - d_max - psi_max
n_max_theory = b_max - d_min  # = b_max

print(f"\nStep 2: (b,l) → (n,m) via m = l + d, n = b - d - ψ(d)")
print(f"  b ∈ [{b_min:.2f}, {b_max:.2f}]")
print(f"  l ∈ [{l_min_theory:.2f}, {l_max_theory:.2f}]")
print(f"  d ∈ [{d_min:.2f}, {d_max:.2f}] (estimated)")
print(f"  → m ∈ [{m_min_theory:.2f}, {m_max_theory:.2f}] (theoretical)")
print(f"  → n ∈ [{n_min_theory:.2f}, {n_max_theory:.2f}] (theoretical)")
print(f"  BUT target m ∈ [{par.grid_m[0]:.2f}, {par.grid_m[-1]:.2f}]")
print(f"  AND target n ∈ [{par.grid_n[0]:.2f}, {par.grid_n[-1]:.2f}]")

if m_max_theory > par.grid_m[-1]:
    print(f"  ⚠ m can reach {m_max_theory:.2f} but target only {par.grid_m[-1]:.2f}")
if n_min_theory < par.grid_n[0]:
    print(f"  ⚠ n can drop to {n_min_theory:.2f} but target starts at {par.grid_n[0]:.2f}")

# Empirical coverage check
print("\n" + "=" * 70)
print("EMPIRICAL COVERAGE FROM ACTUAL SOLUTION")
print("=" * 70)

t = 5  # Mid-life

# Get intermediate values
c_pure_c = segm.sol.c_pure_c[t]
v_l = segm.sol.v_pure_c_l[t]
v_b = segm.sol.v_pure_c_b[t]

print(f"\nSubproblem 1: Pure consumption on (b={par.Nb_pd}, l={len(par.grid_m)}) grid")
print(f"  c: mean={c_pure_c.mean():.3f}, range=[{c_pure_c.min():.3f}, {c_pure_c.max():.3f}]")
print(f"  Implied a = l - c: range=[{(par.grid_m[:, None] - c_pure_c).min():.3f}, {(par.grid_m[:, None] - c_pure_c).max():.3f}]")

# Count how many (b,l) points can generate valid d>0
valid_count = 0
high_d_count = 0

for i_b in range(par.Nb_pd):
    b_val = par.grid_b_pd[i_b]
    for i_l in range(len(par.grid_m)):
        l_val = par.grid_m[i_l]
        
        v_l_val = v_l[i_b, i_l]
        v_b_val = v_b[i_b, i_l]
        c_val = c_pure_c[i_b, i_l]
        
        if c_val > 1e-8:
            denom = v_l_val - v_b_val
            if denom > 1e-6 and v_b_val > 1e-10:
                d_test = (par.chi * v_b_val) / denom - 1.0
                if d_test > 0:
                    valid_count += 1
                    if d_test > 0.5:
                        high_d_count += 1

total_points = par.Nb_pd * len(par.grid_m)
print(f"\nSubproblem 2: Pension FOC can be inverted for d>0 at:")
print(f"  {valid_count}/{total_points} points ({100*valid_count/total_points:.1f}%)")
print(f"  {high_d_count}/{total_points} with d>0.5 ({100*high_d_count/total_points:.1f}%)")

# Check actual endogenous (n,m) coverage
print("\n" + "=" * 70)
print("PROPOSED FIX")
print("=" * 70)
print()

print("Option 1: Extend intermediate grid_l to cover larger liquid resources")
print(f"  Current: l ∈ [0, {par.m_max:.1f}]")
print(f"  Proposed: l ∈ [0, {a_max + c_max:.1f}] to match a_max + c_max")
print(f"  Impact: Allows SEGM to cover full range without exceeding m_max after adding d")
print()

print("Option 2: Extend target grid_m to allow for endogenous m > m_max")
print(f"  Current: m ∈ [0, {par.m_max:.1f}]")
print(f"  Proposed: m ∈ [0, {par.m_max:.1f}] but validation allows m < {par.m_max + 3:.1f}")
print(f"  Impact: Already implemented, seems to help")
print()

print("Option 3: Use different grid_l that's smaller than grid_m")
print(f"  Current: grid_l = grid_m")
print(f"  Proposed: grid_l with max = {par.m_max - 2:.1f} to leave room for d")
print(f"  Impact: Ensures m = l + d stays within [0, {par.m_max + 1:.1f}]")
print()

print("Recommendation:")
print("  Try Option 3: Create grid_l ∈ [0, m_max - 2] so that:")
print("    - l_max + d_max ≈ (m_max - 2) + 2 = m_max")
print("    - Endogenous m stays within target grid range")
print("    - Better alignment between intermediate and target states")
print()

