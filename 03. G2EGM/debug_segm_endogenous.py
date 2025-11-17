"""
Debug SEGM's endogenous grid generation at problem points
"""

import numpy as np
from numba import njit
from G2EGMModel import G2EGMModelClass
from consav import linear_interp
import utility
import pens

print("=" * 70)
print("DEBUG: SEGM Endogenous Grid at Problem Region")
print("=" * 70)
print()

# Solve SEGM
segm = G2EGMModelClass(name='segm', par={'solmethod': 'SEGM', 'sigma': 0.5, 'T': 15, 'do_print': False})
segm.solve()

# Problem point: (n=0, m≈10)
t = 7
n_target = 0.0
m_target = 9.817

par = segm.par
sol = segm.sol

# Get intermediate subproblem 1 results
c_pure_c = sol.c_pure_c[t]
v_l = sol.v_pure_c_l[t]
v_b = sol.v_pure_c_b[t]

# Get post-decision values
w = sol.w[t]
wa = sol.wa[t]
wb = sol.wb[t]

print(f"Target state: (n={n_target:.3f}, m={m_target:.3f})")
print()

# Check what happens in the pension EGM step for low b (≈n) and high l (≈m)
print("Checking pension EGM at low b, high l:")
print("-" * 70)

# Look at b close to 0
i_b_low = 0
i_b_mid = len(par.grid_b_pd) // 4
i_b_high = len(par.grid_b_pd) // 2

# Look at l close to m_max
i_l_low = 0
i_l_mid = len(par.grid_m) // 2
i_l_high = len(par.grid_m) - 1

test_points = [
    ("Low b, High l", i_b_low, i_l_high),
    ("Mid b, High l", i_b_mid, i_l_high),
    ("High b, High l", i_b_high, i_l_high),
    ("Low b, Mid l", i_b_low, i_l_mid),
]

for name, i_b, i_l in test_points:
    b_val = par.grid_b_pd[i_b]
    l_val = par.grid_m[i_l]
    
    c_val = c_pure_c[i_b, i_l]
    v_l_val = v_l[i_b, i_l]
    v_b_val = v_b[i_b, i_l]
    
    print(f"\n{name}: b={b_val:.3f}, l={l_val:.3f}")
    print(f"  Subproblem 1 output:")
    print(f"    c={c_val:.4f}")
    print(f"    v_l={v_l_val:.6f}, v_b={v_b_val:.6f}")
    print(f"    v_l/v_b={v_l_val/v_b_val if v_b_val > 1e-10 else np.inf:.4f}")
    
    if c_val <= 1e-8:
        print(f"  ⚠ Invalid consumption - skipped in pension EGM")
        continue
    
    # Try to invert pension FOC manually
    denom = v_l_val - v_b_val
    
    if denom > 1e-6 and v_b_val > 1e-10:
        d_test = (par.chi * v_b_val) / denom - 1.0
        
        print(f"  FOC inversion:")
        print(f"    v_l - v_b = {denom:.6f}")
        print(f"    d_unconstrained = {d_test:.4f}")
        
        if d_test > 0:
            # Clamp to feasibility
            d_feasible = min(d_test, b_val, l_val)
            print(f"    d_feasible = min({d_test:.2f}, b={b_val:.2f}, l={l_val:.2f}) = {d_feasible:.4f}")
            
            # Compute endogenous (n, m)
            psi_val = par.chi * np.log(1 + d_feasible)
            n_endo = b_val - d_feasible - psi_val
            m_endo = l_val + d_feasible
            a_endo = m_endo - c_val
            
            print(f"  Endogenous states:")
            print(f"    n = b - d - ψ(d) = {b_val:.2f} - {d_feasible:.2f} - {psi_val:.2f} = {n_endo:.4f}")
            print(f"    m = l + d = {l_val:.2f} + {d_feasible:.2f} = {m_endo:.4f}")
            print(f"    a = m - c = {m_endo:.2f} - {c_val:.2f} = {a_endo:.4f}")
            
            # Check if endogenous point is valid
            valid = True
            reasons = []
            
            if a_endo < par.grid_a_pd[0] or a_endo > par.grid_a_pd[-1]:
                valid = False
                reasons.append(f"a out of bounds [{par.grid_a_pd[0]:.2f}, {par.grid_a_pd[-1]:.2f}]")
            
            if n_endo < 0 or n_endo < par.grid_n[0]:
                valid = False
                reasons.append(f"n too low ({n_endo:.4f} < {par.grid_n[0]:.2f})")
            
            if m_endo < par.grid_m[0]:
                valid = False
                reasons.append(f"m too low ({m_endo:.4f} < {par.grid_m[0]:.2f})")
            
            if m_endo > par.m_max + 1:
                valid = False
                reasons.append(f"m too high ({m_endo:.4f} > {par.m_max + 1:.2f})")
            
            if valid:
                print(f"  ✓ Valid endogenous point")
            else:
                print(f"  ✗ Invalid: {', '.join(reasons)}")
        else:
            print(f"    d_unconstrained < 0 → skipped")
    else:
        print(f"  FOC inversion:")
        print(f"    v_l - v_b = {denom:.6f} (≤0 or too small)")
        print(f"    Cannot invert FOC → d=0 only")

# Summary
print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)
print()
print("Expected behavior:")
print("  • At low n, high m: should have high pension savings (d)")
print("  • v_l should be > v_b (liquid wealth more valuable than pension)")
print("  • Endogenous (n,m) should map back to target grid region")
print()
print("Potential issues:")
print("  1. v_l ≈ v_b → can't invert FOC → d=0")
print("  2. Endogenous (n,m) out of target grid bounds")
print("  3. Upper envelope fails to interpolate to target region")
print()

