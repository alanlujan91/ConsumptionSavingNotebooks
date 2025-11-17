"""
Check if post-decision grids are limiting SEGM's performance
"""

import numpy as np
from G2EGMModel import G2EGMModelClass

print("=" * 70)
print("POST-DECISION GRID COVERAGE CHECK")
print("=" * 70)
print()

segm = G2EGMModelClass(name='segm', par={'solmethod': 'SEGM', 'sigma': 0.5, 'T': 10, 'do_print': False})
segm.solve()

par = segm.par
sol = segm.sol

print("Post-decision grids:")
print(f"  a: [{par.grid_a_pd[0]:.2f}, {par.grid_a_pd[-1]:.2f}], N={par.Na_pd}")
print(f"  b: [{par.grid_b_pd[0]:.2f}, {par.grid_b_pd[-1]:.2f}], N={par.Nb_pd}")
print()

# Check what values of (a, b) are implied by the consumption subproblem
t = 5
c_pure = sol.c_pure_c[t]

a_implied = []
b_implied = []

for i_b in range(par.Nb_pd):
    b_val = par.grid_b_pd[i_b]
    for i_l in range(par.Nm):
        l_val = par.grid_m[i_l]
        c_val = c_pure[i_b, i_l]
        
        if c_val > 1e-8:
            a_val = l_val - c_val
            if a_val > 0:
                a_implied.append(a_val)
                b_implied.append(b_val)

a_implied = np.array(a_implied)
b_implied = np.array(b_implied)

print(f"Implied post-decision states from consumption subproblem:")
print(f"  a: [{a_implied.min():.2f}, {a_implied.max():.2f}]")
print(f"  b: [{b_implied.min():.2f}, {b_implied.max():.2f}]")
print()

# Check if these exceed the post-decision grids
a_exceed = (a_implied > par.grid_a_pd[-1]).sum()
b_exceed = (b_implied > par.grid_b_pd[-1]).sum()

if a_exceed > 0:
    print(f"⚠ {a_exceed} points have a > a_max = {par.grid_a_pd[-1]:.2f}")
    print(f"  Max excess: {a_implied.max() - par.grid_a_pd[-1]:.2f}")
else:
    print(f"✓ All a values within post-decision grid")

if b_exceed > 0:
    print(f"⚠ {b_exceed} points have b > b_max = {par.grid_b_pd[-1]:.2f}")
    print(f"  Max excess: {b_implied.max() - par.grid_b_pd[-1]:.2f}")
else:
    print(f"✓ All b values within post-decision grid")

print()

# Now check what (a,b) values are needed for the pension subproblem
print("=" * 70)
print("POST-DECISION STATES NEEDED FOR PENSION SUBPROBLEM")
print("=" * 70)
print()

# At pension subproblem, we evaluate w(b*, a*) where:
# b* = n + d + ψ(d) and a* = m - c - d

# Get final policies
c_final = sol.c[t]
d_final = sol.d[t]

a_needed = []
b_needed = []

for i_n in range(par.Nn):
    n_val = par.grid_n[i_n]
    for i_m in range(par.Nm):
        m_val = par.grid_m[i_m]
        c_val = c_final[i_n, i_m]
        d_val = d_final[i_n, i_m]
        
        if c_val > 0 and d_val >= 0:
            a_val = m_val - c_val - d_val
            psi_val = par.chi * np.log(1 + d_val)
            b_val = n_val + d_val + psi_val
            
            if a_val > 0 and b_val > 0:
                a_needed.append(a_val)
                b_needed.append(b_val)

a_needed = np.array(a_needed)
b_needed = np.array(b_needed)

print(f"Post-decision states needed to evaluate final policies:")
print(f"  a: [{a_needed.min():.2f}, {a_needed.max():.2f}]")
print(f"  b: [{b_needed.min():.2f}, {b_needed.max():.2f}]")
print()

# Check coverage
a_below = (a_needed < par.grid_a_pd[0]).sum()
a_above = (a_needed > par.grid_a_pd[-1]).sum()
b_below = (b_needed < par.grid_b_pd[0]).sum()
b_above = (b_needed > par.grid_b_pd[-1]).sum()

print("Coverage check:")
if a_below > 0:
    print(f"  ⚠ {a_below} points need a < a_min = {par.grid_a_pd[0]:.2f}")
if a_above > 0:
    print(f"  ⚠ {a_above} points need a > a_max = {par.grid_a_pd[-1]:.2f}")
if b_below > 0:
    print(f"  ⚠ {b_below} points need b < b_min = {par.grid_b_pd[0]:.2f}")
if b_above > 0:
    print(f"  ⚠ {b_above} points need b > b_max = {par.grid_b_pd[-1]:.2f}")

if a_below == 0 and a_above == 0 and b_below == 0 and b_above == 0:
    print("  ✓ All points within post-decision grid bounds")

print()
print("=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print()

if a_above > 0 or b_above > 0:
    print("SEGM needs larger post-decision grids:")
    if a_above > 0:
        print(f"  • Extend a_max from {par.grid_a_pd[-1]:.1f} to {np.ceil(a_needed.max()):.1f}")
    if b_above > 0:
        print(f"  • Extend b_max from {par.grid_b_pd[-1]:.1f} to {np.ceil(b_needed.max()):.1f}")
    print()
    print("This allows accurate w(b,a) interpolation without extrapolation")
else:
    print("Post-decision grids are adequate.")
    print("The issue is likely in the upper envelope regridding, not grid coverage.")

print()

