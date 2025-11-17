"""
Test taste shocks (sigma parameter) in G2EGM and NEGM

As sigma increases, the discrete retirement choice becomes smooth,
reducing kinks in value functions and making the upper envelope less critical.
"""

import numpy as np
from G2EGMModel import G2EGMModelClass

print("=" * 70)
print("TASTE SHOCKS TEST - G2EGM vs NEGM")
print("=" * 70)
print()

# Test different sigma values
sigma_values = [0.0, 0.01, 0.05, 0.10, 0.20, 0.50]

for method in ['G2EGM', 'NEGM']:
    print(f"\n{method} with varying taste shocks (sigma):")
    print("-" * 70)
    
    for sigma in sigma_values:
        # Solve model with this sigma
        model = G2EGMModelClass(
            name=f'{method}_sigma{sigma}',
            par={
                'solmethod': method,
                'sigma': sigma,
                'T': 10,  # Shorter horizon for faster testing
                'do_print': False
            }
        )
        model.solve()
        
        # Check solution quality at mid-life
        t = 5
        c = model.sol.c[t]
        d = model.sol.d[t]
        v = -1.0 / model.sol.inv_v[t]  # Convert inv_v back to v
        
        # Measure smoothness: standard deviation of values indicates kinks
        # (More kinks → higher local variation → higher std after detrending)
        v_smooth = v[~np.isinf(v) & ~np.isnan(v)]
        c_smooth = c[c > 0]
        d_smooth = d[d >= 0]
        
        # Compute value function statistics
        v_mean = v_smooth.mean() if len(v_smooth) > 0 else np.nan
        v_std = v_smooth.std() if len(v_smooth) > 0 else np.nan
        v_min = v_smooth.min() if len(v_smooth) > 0 else np.nan
        v_max = v_smooth.max() if len(v_smooth) > 0 else np.nan
        
        # Check consumption and pension choices
        c_mean = c_smooth.mean() if len(c_smooth) > 0 else np.nan
        d_mean = d_smooth.mean() if len(d_smooth) > 0 else np.nan
        d_positive_pct = 100 * (d > 0.01).sum() / d.size
        
        print(f"  σ={sigma:5.2f}: ", end="")
        print(f"c={c_mean:5.3f}, d={d_mean:5.3f} ({d_positive_pct:4.1f}% >0), ", end="")
        print(f"V∈[{v_min:7.2f}, {v_max:7.2f}] (range={v_max-v_min:6.2f})")

print()
print("=" * 70)
print("INTERPRETATION:")
print("=" * 70)
print("As σ increases:")
print("  • Retirement choice becomes smoother (continuous vs discrete)")
print("  • Value function range typically decreases (less variation)")
print("  • Upper envelope becomes less critical (fewer kinks to resolve)")
print("  • Policies may shift slightly as smoothing changes incentives")
print()
print("When σ=0: Hard max, discrete retirement choice (original model)")
print("When σ>0: Smooth choice with taste shocks (logit-style smoothing)")
print()

# Test numerical stability
print("\nNumerical stability check:")
print("-" * 70)

# Very small sigma (should behave like sigma=0)
model_small = G2EGMModelClass(
    name='test_small_sigma',
    par={'solmethod': 'G2EGM', 'sigma': 1e-12, 'T': 5, 'do_print': False}
)
model_small.solve()

# Exactly zero sigma
model_zero = G2EGMModelClass(
    name='test_zero_sigma', 
    par={'solmethod': 'G2EGM', 'sigma': 0.0, 'T': 5, 'do_print': False}
)
model_zero.solve()

# Compare solutions
t = 2
diff_c = np.abs(model_small.sol.c[t] - model_zero.sol.c[t]).max()
diff_d = np.abs(model_small.sol.d[t] - model_zero.sol.d[t]).max()

print(f"σ=1e-12 vs σ=0.0:")
print(f"  Max |Δc| = {diff_c:.6e} (should be ≈0)")
print(f"  Max |Δd| = {diff_d:.6e} (should be ≈0)")
print()

if diff_c < 1e-6 and diff_d < 1e-6:
    print("✓ Numerical stability: PASSED")
else:
    print("✗ Numerical stability: FAILED")

print()

