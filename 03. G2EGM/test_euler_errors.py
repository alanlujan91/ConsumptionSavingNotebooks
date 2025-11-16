"""
Compare Euler errors across NEGM, G2EGM, and manual SEGM
"""

import numpy as np
from G2EGMModel import G2EGMModelClass
import utility
import pens
from consav import linear_interp
import manual_segm

T_test = 20  # Use default T

print("="*70)
print(f"EULER ERROR COMPARISON (T={T_test})")
print("="*70)

# Initialize models
model_g2egm = G2EGMModelClass(name='G2EGM')
model_g2egm.par.solmethod = 'G2EGM'
model_g2egm.par.T = T_test
model_g2egm.solve()

print("\nG2EGM solved successfully")
print(f"  d at t=0: mean={model_g2egm.sol.d[0].mean():.3f}, max={model_g2egm.sol.d[0].max():.3f}")
print(f"  d>0.01: {np.sum(model_g2egm.sol.d[0] > 0.01)} / {model_g2egm.sol.d[0].size}")

# Store results
results = {
    'G2EGM': {
        'c': model_g2egm.sol.c[0],
        'd': model_g2egm.sol.d[0],
    },
}

# Use common parameters
par = model_g2egm.par
grid_n = par.grid_n
grid_m = par.grid_m

# Euler error calculation (simplified for T=1)
def compute_euler_errors(c_pol, d_pol, par, grid_n, grid_m):
    """
    Compute Euler errors for consumption-pension problem
    
    Euler equation: u'(c) = β*Ra*E[u'(c_plus)]
    
    For T=1:
    - c_plus = Ra*a + Rb*b + eta (consume everything next period)
    - a = m - c - d
    - b = n + d + ψ(d)
    """
    
    # Utility functions (pure Python)
    rho = par.rho
    def u_prime_py(c):
        return c**(-rho)
    def inv_u_prime_py(x):
        return x**(-1/rho)
    
    # Pension function (pure Python)
    chi = par.chi
    def psi_py(d):
        return chi * np.log(1 + d)
    
    # Test grid (wider range)
    K = 40
    min_m, max_m = 1.0, 8.0
    min_n, max_n = 0.5, 8.0
    
    n_test = np.linspace(min_n, max_n, K)
    m_test = np.linspace(min_m, max_m, K)
    
    euler_errors = []
    n_skipped_a = 0
    n_skipped_c = 0
    
    for n in n_test:
        for m in m_test:
            # Get policies
            c = linear_interp.interp_2d(grid_n, grid_m, c_pol, n, m)
            d = linear_interp.interp_2d(grid_n, grid_m, d_pol, n, m)
            
            c = np.clip(c, 1e-8, m)
            d = np.clip(d, 0, m - c)
            
            # Compute savings
            a = m - c - d
            b = n + d + psi_py(d)
            
            if a < 1e-6:
                n_skipped_a += 1
                continue
            if c < 1e-8:
                n_skipped_c += 1
                continue
            
            # RHS of Euler equation (expectation over eta shocks)
            RHS = 0.0
            for i_eta in range(par.Neta):
                # Next period total resources (consume everything)
                c_plus = par.Ra * a + par.Rb * b + par.eta[i_eta]
                
                if c_plus > 1e-8:
                    RHS += par.w_eta[i_eta] * par.beta * par.Ra * u_prime_py(c_plus)
            
            # Euler error
            c_implied = inv_u_prime_py(RHS)
            euler_raw = c - c_implied
            euler_rel = np.log10(np.abs(euler_raw / c) + 1e-16)
            
            euler_errors.append(euler_rel)
    
    print(f"  Debug: skipped {n_skipped_a} points (a<1e-6), {n_skipped_c} points (c<1e-8)")
    return np.array(euler_errors)

print("\nComputing Euler errors...\n")

for method in ['G2EGM']:
    c_pol = results[method]['c']
    d_pol = results[method]['d']
    
    euler_errors = compute_euler_errors(c_pol, d_pol, par, grid_n, grid_m)
    
    if len(euler_errors) > 0:
        print(f"{method}:")
        print(f"  Mean Euler error:   {np.mean(euler_errors):8.3f}")
        print(f"  Median Euler error: {np.median(euler_errors):8.3f}")
        print(f"  Max Euler error:    {np.max(euler_errors):8.3f}")
        print(f"  P5 Euler error:     {np.percentile(euler_errors, 5):8.3f}")
        print(f"  P95 Euler error:    {np.percentile(euler_errors, 95):8.3f}")
        print(f"  # valid points:     {len(euler_errors)}")
    else:
        print(f"{method}: No valid points")
    print()

print("="*70)
print("NOTE: Euler errors in log10 scale")
print("  -3 = 0.1% error, -4 = 0.01% error, -5 = 0.001% error")
print("="*70)

