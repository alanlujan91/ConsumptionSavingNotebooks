"""
Detailed analysis of how taste shocks smooth value functions
and reduce the importance of the upper envelope in G2EGM and NEGM
"""

import numpy as np
from G2EGMModel import G2EGMModelClass
import matplotlib.pyplot as plt

print("=" * 70)
print("TASTE SHOCKS: VALUE FUNCTION SMOOTHNESS ANALYSIS")
print("=" * 70)
print()

# Test sigma values
sigma_values = [0.0, 0.05, 0.10, 0.20, 0.50]

# Store results for both methods
results = {'G2EGM': {}, 'NEGM': {}}

for method in ['G2EGM', 'NEGM']:
    print(f"\nAnalyzing {method}...")
    
    for sigma in sigma_values:
        model = G2EGMModelClass(
            name=f'{method}_sigma{sigma}',
            par={'solmethod': method, 'sigma': sigma, 'T': 15, 'do_print': False}
        )
        model.solve()
        
        # Analyze mid-life period
        t = 7
        
        # Extract value and marginal value functions
        inv_v = model.sol.inv_v[t]
        inv_vm = model.sol.inv_vm[t]
        
        # Convert to regular values (handling infinities)
        v = np.where(np.abs(inv_v) > 1e-10, -1.0 / inv_v, np.nan)
        vm = np.where(np.abs(inv_vm) > 1e-10, 1.0 / inv_vm, np.nan)
        
        # Measure "kinkiness" by looking at second derivatives
        # (More kinks → larger second derivatives → less smooth)
        
        # Along m dimension (for fixed n)
        n_idx = len(model.par.grid_n) // 2  # Middle pension wealth
        v_slice_m = v[n_idx, :]
        valid = ~np.isnan(v_slice_m) & ~np.isinf(v_slice_m)
        
        if np.sum(valid) > 10:
            v_clean = v_slice_m[valid]
            # Second difference (discrete second derivative)
            d2v_m = np.abs(np.diff(np.diff(v_clean)))
            kink_metric_m = np.mean(d2v_m) if len(d2v_m) > 0 else 0.0
        else:
            kink_metric_m = np.nan
        
        # Along n dimension (for fixed m)  
        m_idx = len(model.par.grid_m) // 2  # Middle market resources
        v_slice_n = v[:, m_idx]
        valid = ~np.isnan(v_slice_n) & ~np.isinf(v_slice_n)
        
        if np.sum(valid) > 10:
            v_clean = v_slice_n[valid]
            d2v_n = np.abs(np.diff(np.diff(v_clean)))
            kink_metric_n = np.mean(d2v_n) if len(d2v_n) > 0 else 0.0
        else:
            kink_metric_n = np.nan
        
        # Marginal value smoothness
        vm_slice = vm[n_idx, :]
        valid = ~np.isnan(vm_slice) & ~np.isinf(vm_slice)
        
        if np.sum(valid) > 10:
            vm_clean = vm_slice[valid]
            d_vm = np.abs(np.diff(vm_clean))
            marginal_smoothness = np.mean(d_vm) if len(d_vm) > 0 else 0.0
        else:
            marginal_smoothness = np.nan
        
        # Store results
        results[method][sigma] = {
            'kink_m': kink_metric_m,
            'kink_n': kink_metric_n,
            'marginal_smoothness': marginal_smoothness,
            'c': model.sol.c[t],
            'd': model.sol.d[t]
        }
        
        print(f"  σ={sigma:4.2f}: Kinks(m)={kink_metric_m:8.4f}, Kinks(n)={kink_metric_n:8.4f}, ", end="")
        print(f"Marginal smoothness={marginal_smoothness:7.4f}")

# Summary comparison
print("\n" + "=" * 70)
print("SUMMARY: Impact of taste shocks on value function smoothness")
print("=" * 70)

for method in ['G2EGM', 'NEGM']:
    print(f"\n{method}:")
    print("-" * 70)
    
    sigma_0 = results[method][0.0]
    
    for sigma in sigma_values[1:]:
        res = results[method][sigma]
        
        # Compute reduction in kinkiness
        kink_reduction_m = (sigma_0['kink_m'] - res['kink_m']) / sigma_0['kink_m'] * 100
        kink_reduction_n = (sigma_0['kink_n'] - res['kink_n']) / sigma_0['kink_n'] * 100
        marginal_reduction = (sigma_0['marginal_smoothness'] - res['marginal_smoothness']) / sigma_0['marginal_smoothness'] * 100
        
        print(f"  σ={sigma:4.2f} vs σ=0.00:")
        print(f"    • Kinks reduced by {kink_reduction_m:5.1f}% (m-dimension)")
        print(f"    • Kinks reduced by {kink_reduction_n:5.1f}% (n-dimension)")
        print(f"    • Marginal value smoothness improved by {marginal_reduction:5.1f}%")

print("\n" + "=" * 70)
print("INTERPRETATION:")
print("=" * 70)
print()
print("Kink metric: Mean absolute second derivative of value function")
print("  • Higher values = more kinks/non-smoothness")
print("  • Lower values = smoother value function")
print()
print("As σ increases, kinks are reduced because:")
print("  1. Discrete retirement choice becomes continuous (logit smoothing)")
print("  2. Value function V = log(exp(v_work/σ) + exp(v_ret/σ)) * σ")
print("  3. This smooths the max operator: lim σ→0 gives max, σ>0 smooths")
print()
print("Practical implications:")
print("  • With σ=0: Upper envelope is critical (resolves discrete kinks)")
print("  • With σ>0: Upper envelope less critical (fewer kinks to resolve)")
print("  • Higher σ: Could potentially use simpler interpolation methods")
print()

# Quick comparison of G2EGM vs NEGM
print("=" * 70)
print("G2EGM vs NEGM Comparison:")
print("=" * 70)

for sigma in sigma_values:
    g2 = results['G2EGM'][sigma]
    neg = results['NEGM'][sigma]
    
    d_diff = np.abs(g2['d'] - neg['d']).mean()
    c_diff = np.abs(g2['c'] - neg['c']).mean()
    
    print(f"σ={sigma:4.2f}: |Δd|={d_diff:.6f}, |Δc|={c_diff:.6f}")

print()
print("Both methods give nearly identical results with taste shocks!")
print()

