"""1-Period SEGM Test - No Numba, Pure Python"""

import numpy as np
from scipy.interpolate import interp1d

# Simple utility function
def u(c, rho=2.0):
    """CRRA utility"""
    return c**(1-rho) / (1-rho)

def u_prime(c, rho=2.0):
    """Marginal utility"""
    return c**(-rho)

def u_prime_inv(uc, rho=2.0):
    """Inverse marginal utility"""
    return uc**(-1/rho)

def psi(d, chi=1.5):
    """Pension subsidy function"""
    return chi * np.log(1 + d)

print("="*70)
print("1-PERIOD SEGM TEST")
print("="*70)

# Parameters - MATCH THE ACTUAL MODEL
beta = 0.98
rho = 2.0
Ra = 1.02   # Return on liquid assets (lower)
Rb = 1.04   # Return on pension assets (HIGHER - this is the key!)
chi = 0.10  # Pension subsidy (MUCH smaller than I was using!)

# Simple grids
grid_a = np.linspace(0.01, 5, 10)  # Post-decision liquid assets
grid_b = np.linspace(0.01, 5, 8)   # Post-decision pension wealth

grid_n = np.linspace(0.01, 5, 12)  # Pre-decision pension wealth
grid_m = np.linspace(0.01, 8, 15)  # Pre-decision total resources

print(f"\nParameters:")
print(f"  χ = {chi}, β = {beta}, ρ = {rho}")
print(f"  Ra = {Ra}, Rb = {Rb}")
print(f"  grid_a: {len(grid_a)} points [{grid_a[0]:.2f}, {grid_a[-1]:.2f}]")
print(f"  grid_b: {len(grid_b)} points [{grid_b[0]:.2f}, {grid_b[-1]:.2f}]")

# Step 0: Next period value function (terminal: consume everything)
# V(m, n) = u(m + n)
def V_next(m, n):
    return u(m + n, rho)

# Step 1: Post-decision value function w(b, a)
print("\n" + "-"*70)
print("STEP 1: Post-Decision Value Function w(b, a)")
print("-"*70)

w_grid = np.zeros((len(grid_b), len(grid_a)))
wa_grid = np.zeros((len(grid_b), len(grid_a)))
wb_grid = np.zeros((len(grid_b), len(grid_a)))

for i_b, b in enumerate(grid_b):
    for i_a, a in enumerate(grid_a):
        # Next period states with DIFFERENT RETURNS
        m_next = Ra * a  # Liquid assets earn Ra
        n_next = Rb * b  # Pension assets earn Rb
        
        w_grid[i_b, i_a] = beta * V_next(m_next, n_next)
        
        # Marginal values (envelope theorem on V_next)
        c_next = m_next + n_next
        uc_next = u_prime(c_next, rho)
        
        # Chain rule: ∂w/∂a = β * ∂V/∂m * ∂m/∂a = β * uc * Ra
        wa_grid[i_b, i_a] = beta * Ra * uc_next  # ∂w/∂a
        wb_grid[i_b, i_a] = beta * Rb * uc_next  # ∂w/∂b

print(f"w(b, a) computed: shape {w_grid.shape}")
print(f"  w range: [{w_grid.min():.4f}, {w_grid.max():.4f}]")
print(f"  wa range: [{wa_grid.min():.4f}, {wa_grid.max():.4f}]")
print(f"  wb range: [{wb_grid.min():.4f}, {wb_grid.max():.4f}]")

# Step 2: SUBPROBLEM 1 - Pure Consumption EGM
print("\n" + "-"*70)
print("STEP 2: SUBPROBLEM 1 - Pure Consumption EGM")
print("-"*70)

# For each b, do EGM over a
c_pure_c = {}
v_pure_c = {}
v_l_pure = {}
v_b_pure = {}
grid_l = {}

for i_b, b in enumerate(grid_b):
    print(f"\nProcessing b = {b:.3f}")
    
    # Endogenous grid construction
    c_endo = np.zeros(len(grid_a))
    l_endo = np.zeros(len(grid_a))  # LIQUID resources = a + c
    w_endo = np.zeros(len(grid_a))
    wa_endo = np.zeros(len(grid_a))
    wb_endo = np.zeros(len(grid_a))
    
    for i_a, a in enumerate(grid_a):
        # Invert consumption Euler equation
        c_endo[i_a] = u_prime_inv(wa_grid[i_b, i_a], rho)
        l_endo[i_a] = a + c_endo[i_a]  # Liquid resources
        w_endo[i_a] = w_grid[i_b, i_a]
        wa_endo[i_a] = wa_grid[i_b, i_a]
        wb_endo[i_a] = wb_grid[i_b, i_a]
    
    print(f"  Endogenous l: [{l_endo.min():.2f}, {l_endo.max():.2f}]")
    
    # Upper envelope: regrid to exogenous l grid
    # Simple version: just interpolate (assuming monotone for now)
    # In practice would need proper upper envelope
    
    # Use grid_m as our liquid resources grid
    l_target = grid_m[grid_m <= l_endo.max()]
    
    if len(l_target) == 0:
        print(f"  WARNING: No valid target l points")
        continue
    
    # Interpolate policies and values
    c_interp = interp1d(l_endo, c_endo, bounds_error=False, fill_value='extrapolate')
    v_interp_vals = u(c_endo, rho) + w_endo
    v_interp = interp1d(l_endo, v_interp_vals, bounds_error=False, fill_value='extrapolate')
    
    c_pure_c[b] = c_interp(l_target)
    v_pure_c[b] = v_interp(l_target)
    grid_l[b] = l_target
    
    # ENVELOPE THEOREM: Marginal values at optimal choices
    # v_l(b, l) = w_a(b, a*) where a* = l - c*(b, l)
    # v_b(b, l) = w_b(b, a*)
    
    a_star = l_target - c_pure_c[b]
    
    # Interpolate wa and wb at a*
    wa_at_a_star = interp1d(grid_a, wa_endo, bounds_error=False, fill_value='extrapolate')(a_star)
    wb_at_a_star = interp1d(grid_a, wb_endo, bounds_error=False, fill_value='extrapolate')(a_star)
    
    v_l_pure[b] = wa_at_a_star
    v_b_pure[b] = wb_at_a_star
    
    print(f"  Output l grid: {len(l_target)} points")
    print(f"  c: [{c_pure_c[b].min():.2f}, {c_pure_c[b].max():.2f}]")
    print(f"  v_l: [{v_l_pure[b].min():.4f}, {v_l_pure[b].max():.4f}]")
    print(f"  v_b: [{v_b_pure[b].min():.4f}, {v_b_pure[b].max():.4f}]")
    
    ratio = v_l_pure[b] / v_b_pure[b]
    print(f"  v_l/v_b: [{ratio.min():.3f}, {ratio.max():.3f}]")
    print(f"  Need (1, {chi+1:.2f}) for d>0")
    in_range = (ratio > 1) & (ratio < chi + 1)
    print(f"  Points in valid range: {np.sum(in_range)}/{len(ratio)}")

# Step 3: SUBPROBLEM 2 - Pure Pension EGM
print("\n" + "-"*70)
print("STEP 3: SUBPROBLEM 2 - Pure Pension EGM")
print("-"*70)

# For each (b, l) from subproblem 1, invert pension FOC
d_results = []
n_results = []
m_results = []
c_results = []
v_results = []

for b in grid_b:
    if b not in grid_l:
        continue
    
    l_vals = grid_l[b]
    c_vals = c_pure_c[b]
    v_vals = v_pure_c[b]
    v_l_vals = v_l_pure[b]
    v_b_vals = v_b_pure[b]
    
    print(f"\nProcessing b = {b:.3f}, {len(l_vals)} liquid resource points")
    
    for i_l, l in enumerate(l_vals):
        v_l_val = v_l_vals[i_l]
        v_b_val = v_b_vals[i_l]
        c_val = c_vals[i_l]
        v_val = v_vals[i_l]
        
        # Invert pension FOC
        denom = v_l_val - v_b_val
        
        if v_b_val > 1e-10 and denom > 1e-10:
            # d = χ * v_b / (v_l - v_b) - 1
            ratio_chi_vb = chi * v_b_val
            d = ratio_chi_vb / denom - 1.0
            
            # Ensure d >= 0 (critical for log(1 + d))
            d = max(0.0, d)
            d = min(d, min(b, l, 100.0))  # Cap at reasonable max
        else:
            d = 0.0
        
        # Compute endogenous states
        psi_d = psi(d, chi)
        n_endo = b - d - psi_d
        m_endo = l + d
        
        d_results.append(d)
        n_results.append(n_endo)
        m_results.append(m_endo)
        c_results.append(c_val)
        v_results.append(v_val)
    
    # Sample print
    if len(d_results) > 0:
        recent = d_results[-5:]
        print(f"  Last 5 d values: {[f'{x:.4f}' for x in recent]}")

d_results = np.array(d_results)
n_results = np.array(n_results)
m_results = np.array(m_results)

print(f"\nSEGM Results:")
print(f"  Total grid points: {len(d_results)}")
print(f"  d: min={d_results.min():.4f}, max={d_results.max():.4f}, mean={d_results.mean():.4f}")
print(f"  Nonzero d (>0.01): {np.sum(d_results > 0.01)}")
print(f"  n: min={n_results.min():.2f}, max={n_results.max():.2f}")
print(f"  m: min={m_results.min():.2f}, max={m_results.max():.2f}")

# Print some sample points with details
print(f"\nSample points where d might be positive:")
for i in np.argsort(-d_results)[:5]:
    print(f"  d={d_results[i]:.4f}, n={n_results[i]:.2f}, m={m_results[i]:.2f}, c={c_results[i]:.2f}")

print("\n" + "="*70)
print("TEST COMPLETED")
print("="*70)

