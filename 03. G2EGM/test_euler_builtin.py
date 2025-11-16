"""
Test Euler errors using the built-in calculate_euler method
"""

import numpy as np
from G2EGMModel import G2EGMModelClass

print("="*70)
print("EULER ERROR COMPARISON (Built-in method)")
print("="*70)

# G2EGM
print("\nSolving G2EGM...")
model_g2egm = G2EGMModelClass(name='G2EGM', par={'solmethod': 'G2EGM', 'T': 20})
try:
    model_g2egm.solve()
    model_g2egm.calculate_euler()
    
    euler_g2egm = model_g2egm.sim.euler
    
    print(f"✓ G2EGM solved and Euler errors computed")
    print(f"  Mean Euler error:   {np.nanmean(euler_g2egm):8.3f}")
    print(f"  Median Euler error: {np.nanmedian(euler_g2egm):8.3f}")
    print(f"  Max Euler error:    {np.nanmax(euler_g2egm):8.3f}")
    print(f"  P5 Euler error:     {np.nanpercentile(euler_g2egm, 5):8.3f}")
    print(f"  P95 Euler error:    {np.nanpercentile(euler_g2egm, 95):8.3f}")
    print(f"  Non-NaN points:     {np.sum(~np.isnan(euler_g2egm))}")
except Exception as e:
    print(f"✗ G2EGM failed: {e}")
    import traceback
    traceback.print_exc()

# NEGM
print("\nSolving NEGM...")
model_negm = G2EGMModelClass(name='NEGM', par={'solmethod': 'NEGM', 'T': 20})
try:
    model_negm.solve()
    model_negm.calculate_euler()
    
    euler_negm = model_negm.sim.euler
    
    print(f"✓ NEGM solved and Euler errors computed")
    print(f"  Mean Euler error:   {np.nanmean(euler_negm):8.3f}")
    print(f"  Median Euler error: {np.nanmedian(euler_negm):8.3f}")
    print(f"  Max Euler error:    {np.nanmax(euler_negm):8.3f}")
    print(f"  P5 Euler error:     {np.nanpercentile(euler_negm, 5):8.3f}")
    print(f"  P95 Euler error:    {np.nanpercentile(euler_negm, 95):8.3f}")
    print(f"  Non-NaN points:     {np.sum(~np.isnan(euler_negm))}")
except Exception as e:
    print(f"✗ NEGM failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("NOTE: Euler errors in log10 scale")
print("  -3 = 0.1% error, -4 = 0.01% error, -5 = 0.001% error")
print("="*70)

