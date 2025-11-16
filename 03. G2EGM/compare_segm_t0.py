"""Compare manual SEGM with G2EGM and NEGM at t=0 using fresh upstream code"""

import numpy as np
from G2EGMModel import G2EGMModelClass
from scipy.interpolate import RegularGridInterpolator, griddata

print("="*70)
print("SEGM vs G2EGM vs NEGM at t=0")
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
        'inv_v': model.sol.inv_v[t].copy(),
        'w': model.sol.w[t].copy(),
        'wa': model.sol.wa[t].copy() if hasattr(model.sol, 'wa') and len(model.sol.wa) > 0 else None,
        'wb': model.sol.wb[t].copy() if hasattr(model.sol, 'wb') and len(model.sol.wb) > 0 else None,
        'grid_n': model.par.grid_n.copy(),
        'grid_m': model.par.grid_m.copy(),
        'grid_a_pd': model.par.grid_a_pd.copy(),
        'grid_b_pd': model.par.grid_b_pd.copy(),
        'par': model.par,
    }
    
    d = results[method]['d']
    print(f"  ✓ Solved (T={model.par.T})")
    print(f"  d[t=0]: mean={d.mean():.3f}, max={d.max():.3f}, positive={np.sum(d > 0.01)}/{d.size}")

# Check agreement between G2EGM and NEGM
diff_d = np.abs(results['G2EGM']['d'] - results['NEGM']['d'])
diff_c = np.abs(results['G2EGM']['c'] - results['NEGM']['c'])
print(f"\nG2EGM vs NEGM at t=0:")
print(f"  |Δd|: max={diff_d.max():.6f}, mean={diff_d.mean():.6f}")
print(f"  |Δc|: max={diff_c.max():.6f}, mean={diff_c.mean():.6f}")

# Extract common parameters
par = results['G2EGM']['par']
grid_n = results['G2EGM']['grid_n']
grid_m = results['G2EGM']['grid_m']
grid_a_pd = results['G2EGM']['grid_a_pd']
grid_b_pd = results['G2EGM']['grid_b_pd']
w = results['G2EGM']['w']

Ra = par.Ra
Rb = par.Rb
chi = par.chi
beta = par.beta
rho = par.rho

print(f"\n{'='*70}")
print("Manual SEGM Implementation")
print(f"{'='*70}")
print(f"Parameters: Ra={Ra:.4f}, Rb={Rb:.4f}, chi={chi:.4f}, beta={beta:.4f}, rho={rho:.4f}")
print(f"Grids: n={len(grid_n)}, m={len(grid_m)}, a_pd={len(grid_a_pd)}, b_pd={len(grid_b_pd)}")

# Utility functions
def u(c):
    return c**(1-rho) / (1-rho)

def u_prime(c):
    return c**(-rho)

def inv_u_prime(x):
    return x**(-1/rho)

def psi_d(d):
    return chi * np.log(1 + d)

# Compute marginal values wa, wb from w
wa = np.zeros_like(w)
wb = np.zeros_like(w)

for i_b in range(len(grid_b_pd)):
    for i_a in range(len(grid_a_pd)):
        if i_a < len(grid_a_pd) - 1:
            wa[i_b, i_a] = (w[i_b, i_a+1] - w[i_b, i_a]) / (grid_a_pd[i_a+1] - grid_a_pd[i_a])
        else:
            wa[i_b, i_a] = wa[i_b, i_a-1]

for i_a in range(len(grid_a_pd)):
    for i_b in range(len(grid_b_pd)):
        if i_b < len(grid_b_pd) - 1:
            wb[i_b, i_a] = (w[i_b+1, i_a] - w[i_b, i_a]) / (grid_b_pd[i_b+1] - grid_b_pd[i_b])
        else:
            wb[i_b, i_a] = wb[i_b-1, i_a]

print(f"Computed marginals: wa ∈ [{wa.min():.4f}, {wa.max():.4f}], wb ∈ [{wb.min():.4f}, {wb.max():.4f}]")

wb_interp = RegularGridInterpolator((grid_b_pd, grid_a_pd), wb,
                                    bounds_error=False, fill_value=None)

# SUBPROBLEM 1: Pure Consumption (b, a) → (b, l)
print(f"\nSubproblem 1: Pure Consumption...")

c_endo_all = []
l_endo_all = []
b_endo_all = []

for i_b, b_val in enumerate(grid_b_pd):
    for i_a, a_val in enumerate(grid_a_pd):
        wa_val = wa[i_b, i_a]
        if wa_val > 1e-10:
            c_val = inv_u_prime(beta * Ra * wa_val)
            l_val = a_val + c_val  # Liquid resources
            
            c_endo_all.append(c_val)
            l_endo_all.append(l_val)
            b_endo_all.append(b_val)

c_endo_all = np.array(c_endo_all)
l_endo_all = np.array(l_endo_all)
b_endo_all = np.array(b_endo_all)

# Regrid to (b×, l×) where l× = grid_m
c_pure_c = np.zeros((len(grid_b_pd), len(grid_m)))
v_pure_c_l = np.zeros((len(grid_b_pd), len(grid_m)))
v_pure_c_b = np.zeros((len(grid_b_pd), len(grid_m)))

for i_b, b_val in enumerate(grid_b_pd):
    mask = (b_endo_all == b_val)
    l_slice = l_endo_all[mask]
    c_slice = c_endo_all[mask]
    
    if len(l_slice) > 1:
        sort_idx = np.argsort(l_slice)
        l_slice = l_slice[sort_idx]
        c_slice = c_slice[sort_idx]
        
        # Interpolate consumption to grid_m
        c_pure_c[i_b, :] = np.interp(grid_m, l_slice, c_slice, left=0, right=c_slice[-1])
        
        # Compute marginal values using envelope theorem
        for i_l, l_val in enumerate(grid_m):
            c_val = c_pure_c[i_b, i_l]
            a_val = l_val - c_val
            
            # v_l = u'(c)
            v_pure_c_l[i_b, i_l] = u_prime(c_val) if c_val > 1e-10 else 1e10
            
            # v_b = wb(b, a)
            a_clamp = np.clip(a_val, grid_a_pd[0], grid_a_pd[-1])
            v_pure_c_b[i_b, i_l] = wb_interp([[b_val, a_clamp]])[0]

print(f"  c_pure_c: [{c_pure_c.min():.4f}, {c_pure_c.max():.4f}]")
print(f"  v_l: [{v_pure_c_l.min():.4f}, {v_pure_c_l.max():.4f}]")
print(f"  v_b: [{v_pure_c_b.min():.4f}, {v_pure_c_b.max():.4f}]")

# SUBPROBLEM 2: Pure Pension (b, l) → (n, m)
print(f"\nSubproblem 2: Pure Pension...")

d_endo_all = []
c_endo_all2 = []
n_endo_all = []
m_endo_all = []

for i_b, b_val in enumerate(grid_b_pd):
    for i_l, l_val in enumerate(grid_m):
        c_val = c_pure_c[i_b, i_l]
        v_l = v_pure_c_l[i_b, i_l]
        v_b = v_pure_c_b[i_b, i_l]
        
        # FOC for d: ψ'(d) = v_l - v_b
        # chi/(1+d) = v_l - v_b
        # d = chi/(v_l - v_b) - 1
        
        denom = v_l - v_b
        
        if denom > 1e-6 and v_b > 1e-10:
            d_val = chi / denom - 1
            
            # Enforce d >= 0 and feasibility
            if d_val < 0:
                d_val = 0.0
            else:
                d_val = min(d_val, b_val, l_val)
            
            # Compute endogenous (n, m)
            psi_val = psi_d(d_val)
            n_val = b_val - d_val - psi_val
            m_val = l_val + d_val
            
            if n_val >= grid_n[0] and m_val >= grid_m[0]:
                d_endo_all.append(d_val)
                c_endo_all2.append(c_val)
                n_endo_all.append(n_val)
                m_endo_all.append(m_val)

d_endo_all = np.array(d_endo_all)
c_endo_all2 = np.array(c_endo_all2)
n_endo_all = np.array(n_endo_all)
m_endo_all = np.array(m_endo_all)

print(f"  Endogenous points: {len(d_endo_all)}")
print(f"  d_endo: [{d_endo_all.min():.4f}, {d_endo_all.max():.4f}], mean={d_endo_all.mean():.3f}")
print(f"  d_endo>0.01: {np.sum(d_endo_all > 0.01)} / {len(d_endo_all)}")

# Regrid to (n×, m×) using 2D interpolation
if len(d_endo_all) > 0:
    points = np.column_stack([n_endo_all, m_endo_all])
    target_points = np.array([[n, m] for n in grid_n for m in grid_m])

    c_segm = griddata(points, c_endo_all2, target_points, method='linear', fill_value=0)
    d_segm = griddata(points, d_endo_all, target_points, method='linear', fill_value=0)

    c_segm = c_segm.reshape(len(grid_n), len(grid_m))
    d_segm = d_segm.reshape(len(grid_n), len(grid_m))
else:
    c_segm = np.zeros((len(grid_n), len(grid_m)))
    d_segm = np.zeros((len(grid_n), len(grid_m)))

print(f"\n✓ Manual SEGM complete")
print(f"  c_segm: [{c_segm.min():.4f}, {c_segm.max():.4f}]")
print(f"  d_segm: [{d_segm.min():.4f}, {d_segm.max():.4f}], mean={d_segm.mean():.3f}")
print(f"  d_segm>0.01: {np.sum(d_segm > 0.01)} / {d_segm.size}")

# COMPARISON
print(f"\n{'='*70}")
print("COMPARISON")
print(f"{'='*70}")

for method in ['NEGM', 'G2EGM']:
    c_ref = results[method]['c']
    d_ref = results[method]['d']
    
    diff_c = np.abs(c_ref - c_segm)
    diff_d = np.abs(d_ref - d_segm)
    
    print(f"\n{method} vs Manual SEGM:")
    print(f"  |Δc|: max={diff_c.max():.6f}, mean={diff_c.mean():.6f}")
    print(f"  |Δd|: max={diff_d.max():.6f}, mean={diff_d.mean():.6f}")
    
    if diff_c.max() < 0.01 and diff_d.max() < 0.01:
        print(f"  ✓ AGREE (max diff < 0.01)")
    else:
        print(f"  ⚠ DIFFER")
        i_max_d = np.unravel_index(diff_d.argmax(), diff_d.shape)
        i_max_c = np.unravel_index(diff_c.argmax(), diff_c.shape)
        
        print(f"  Max d diff at (n={grid_n[i_max_d[0]]:.2f}, m={grid_m[i_max_d[1]]:.2f}):")
        print(f"    {method}: c={c_ref[i_max_d]:.4f}, d={d_ref[i_max_d]:.4f}")
        print(f"    SEGM:  c={c_segm[i_max_d]:.4f}, d={d_segm[i_max_d]:.4f}")

print(f"\n{'='*70}")

