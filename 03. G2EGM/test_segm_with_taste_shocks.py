"""
Diagnostic test: Compare G2EGM, NEGM, and SEGM with high taste shocks (σ=0.5)

With σ=0.5, the retirement choice is very smooth, so all three methods
should produce nearly identical policies. Any differences indicate bugs in SEGM.
"""

import numpy as np
from G2EGMModel import G2EGMModelClass

print("=" * 70)
print("SEGM DIAGNOSTIC: Comparison under smooth value functions (σ=0.5)")
print("=" * 70)
print()

# Solve all three methods with high taste shocks
sigma = 0.5
methods = ['G2EGM', 'NEGM', 'SEGM']
models = {}

print("Solving all three methods with σ=0.5...")
print("-" * 70)

for method in methods:
    print(f"  Solving {method}...", end=" ")
    try:
        model = G2EGMModelClass(
            name=f'{method}_sigma{sigma}',
            par={
                'solmethod': method,
                'sigma': sigma,
                'T': 15,
                'do_print': False
            }
        )
        model.solve()
        models[method] = model
        print("✓")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()

print()

if len(models) < 3:
    print("Cannot proceed - not all methods solved successfully")
    exit(1)

# Compare solutions at multiple time periods
print("=" * 70)
print("POLICY COMPARISON (pointwise differences)")
print("=" * 70)

time_periods = [5, 7, 10]

for t in time_periods:
    print(f"\nPeriod t={t}:")
    print("-" * 70)
    
    # Extract policies
    c_g2 = models['G2EGM'].sol.c[t]
    d_g2 = models['G2EGM'].sol.d[t]
    
    c_negm = models['NEGM'].sol.c[t]
    d_negm = models['NEGM'].sol.d[t]
    
    c_segm = models['SEGM'].sol.c[t]
    d_segm = models['SEGM'].sol.d[t]
    
    # Compute differences
    print("\nConsumption (c):")
    print(f"  G2EGM:  mean={c_g2.mean():.4f}, std={c_g2.std():.4f}, range=[{c_g2.min():.3f}, {c_g2.max():.3f}]")
    print(f"  NEGM:   mean={c_negm.mean():.4f}, std={c_negm.std():.4f}, range=[{c_negm.min():.3f}, {c_negm.max():.3f}]")
    print(f"  SEGM:   mean={c_segm.mean():.4f}, std={c_segm.std():.4f}, range=[{c_segm.min():.3f}, {c_segm.max():.3f}]")
    
    diff_c_g2_negm = np.abs(c_g2 - c_negm)
    diff_c_g2_segm = np.abs(c_g2 - c_segm)
    diff_c_negm_segm = np.abs(c_negm - c_segm)
    
    print(f"\n  |G2EGM - NEGM|:  mean={diff_c_g2_negm.mean():.6f}, max={diff_c_g2_negm.max():.6f}")
    print(f"  |G2EGM - SEGM|:  mean={diff_c_g2_segm.mean():.6f}, max={diff_c_g2_segm.max():.6f}")
    print(f"  |NEGM - SEGM|:   mean={diff_c_negm_segm.mean():.6f}, max={diff_c_negm_segm.max():.6f}")
    
    print("\nPension deposit (d):")
    print(f"  G2EGM:  mean={d_g2.mean():.4f}, std={d_g2.std():.4f}, range=[{d_g2.min():.3f}, {d_g2.max():.3f}]")
    print(f"  NEGM:   mean={d_negm.mean():.4f}, std={d_negm.std():.4f}, range=[{d_negm.min():.3f}, {d_negm.max():.3f}]")
    print(f"  SEGM:   mean={d_segm.mean():.4f}, std={d_segm.std():.4f}, range=[{d_segm.min():.3f}, {d_segm.max():.3f}]")
    
    diff_d_g2_negm = np.abs(d_g2 - d_negm)
    diff_d_g2_segm = np.abs(d_g2 - d_segm)
    diff_d_negm_segm = np.abs(d_negm - d_segm)
    
    print(f"\n  |G2EGM - NEGM|:  mean={diff_d_g2_negm.mean():.6f}, max={diff_d_g2_negm.max():.6f}")
    print(f"  |G2EGM - SEGM|:  mean={diff_d_g2_segm.mean():.6f}, max={diff_d_g2_segm.max():.6f}")
    print(f"  |NEGM - SEGM|:   mean={diff_d_negm_segm.mean():.6f}, max={diff_d_negm_segm.max():.6f}")
    
    # Identify problematic regions
    threshold = 0.01
    problem_points_c = (diff_c_negm_segm > threshold).sum()
    problem_points_d = (diff_d_negm_segm > threshold).sum()
    
    if problem_points_c > 0:
        print(f"\n  ⚠ {problem_points_c} points with |c_NEGM - c_SEGM| > {threshold}")
    if problem_points_d > 0:
        print(f"  ⚠ {problem_points_d} points with |d_NEGM - d_SEGM| > {threshold}")

# Euler error comparison
print("\n" + "=" * 70)
print("EULER ERROR COMPARISON")
print("=" * 70)

for method in methods:
    try:
        models[method].calculate_euler()
        euler = models[method].sim.euler
        
        print(f"\n{method}:")
        print(f"  Mean:   {np.nanmean(euler):8.3f}")
        print(f"  Median: {np.nanmedian(euler):8.3f}")
        print(f"  P5:     {np.nanpercentile(euler, 5):8.3f}")
        print(f"  P95:    {np.nanpercentile(euler, 95):8.3f}")
        print(f"  Max:    {np.nanmax(euler):8.3f}")
    except Exception as e:
        print(f"\n{method}: Failed to calculate Euler errors - {e}")

# Post-decision value comparison
print("\n" + "=" * 70)
print("POST-DECISION VALUE FUNCTION COMPARISON")
print("=" * 70)

t = 7
print(f"\nPeriod t={t}:")

wa_g2 = models['G2EGM'].sol.wa[t]
wb_g2 = models['G2EGM'].sol.wb[t]

wa_negm = models['NEGM'].sol.wa[t]
wb_negm = models['NEGM'].sol.wb[t] if models['NEGM'].sol.wb.size > 0 else None

wa_segm = models['SEGM'].sol.wa[t]
wb_segm = models['SEGM'].sol.wb[t] if models['SEGM'].sol.wb.size > 0 else None

print(f"\nMarginal value wa (liquid assets):")
print(f"  G2EGM:  mean={wa_g2.mean():.4f}, std={wa_g2.std():.4f}")
print(f"  NEGM:   mean={wa_negm.mean():.4f}, std={wa_negm.std():.4f}")
print(f"  SEGM:   mean={wa_segm.mean():.4f}, std={wa_segm.std():.4f}")

diff_wa = np.abs(wa_negm - wa_segm)
print(f"\n  |wa_NEGM - wa_SEGM|: mean={diff_wa.mean():.6f}, max={diff_wa.max():.6f}")

if wb_segm is not None and wb_negm is not None:
    print(f"\nMarginal value wb (pension wealth):")
    print(f"  G2EGM:  mean={wb_g2.mean():.4f}, std={wb_g2.std():.4f}")
    print(f"  NEGM:   mean={wb_negm.mean():.4f}, std={wb_negm.std():.4f}")
    print(f"  SEGM:   mean={wb_segm.mean():.4f}, std={wb_segm.std():.4f}")
    
    diff_wb = np.abs(wb_negm - wb_segm)
    print(f"\n  |wb_NEGM - wb_SEGM|: mean={diff_wb.mean():.6f}, max={diff_wb.max():.6f}")

# Summary and diagnosis
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)
print()

# Compare max differences
t = 7
c_diff = np.abs(models['NEGM'].sol.c[t] - models['SEGM'].sol.c[t]).max()
d_diff = np.abs(models['NEGM'].sol.d[t] - models['SEGM'].sol.d[t]).max()

threshold_good = 0.001
threshold_ok = 0.01

print("Maximum policy differences (NEGM vs SEGM):")
print(f"  Consumption: {c_diff:.6f}", end="")
if c_diff < threshold_good:
    print("  ✓ Excellent")
elif c_diff < threshold_ok:
    print("  ⚠ Acceptable")
else:
    print("  ✗ Too large - indicates bug")

print(f"  Pension:     {d_diff:.6f}", end="")
if d_diff < threshold_good:
    print("  ✓ Excellent")
elif d_diff < threshold_ok:
    print("  ⚠ Acceptable")
else:
    print("  ✗ Too large - indicates bug")

print()
print("Expected behavior with σ=0.5:")
print("  • All three methods should give nearly identical policies")
print("  • Maximum differences should be < 0.01 (1%)")
print("  • Euler errors should be similar across methods")
print()

if c_diff < threshold_ok and d_diff < threshold_ok:
    print("✓ SEGM appears to be working correctly!")
else:
    print("✗ SEGM has issues - investigate:")
    print("  1. Upper envelope regridding (check triangulation)")
    print("  2. FOC inversion (verify v_l and v_b computation)")
    print("  3. Post-decision value function (ensure wb is computed)")

print()

