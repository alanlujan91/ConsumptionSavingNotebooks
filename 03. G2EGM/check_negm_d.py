"""Check what d values NEGM actually produces"""

import numpy as np
from G2EGMModel import G2EGMModelClass

print("="*70)
print("Checking NEGM pension contributions")
print("="*70)

# Create model with NEGM
model = G2EGMModelClass(name='NEGM')
model.par.solmethod = 'NEGM'
model.par.do_print = False

print(f"\nModel parameters:")
print(f"  Ra = {model.par.Ra}")
print(f"  Rb = {model.par.Rb}")
print(f"  χ = {model.par.chi}")
print(f"  β = {model.par.beta}")
print(f"  ρ = {model.par.rho}")

print(f"\nFor d > 0, need: v_l/v_b in (1, {model.par.chi + 1:.2f})")
print(f"But Rb ({model.par.Rb}) > Ra ({model.par.Ra}), so wb > wa...")
print(f"Which would imply v_b > v_l, i.e., ratio < 1...")

print("\nSolving...")
model.solve()

# Check d values
print("\n" + "-"*70)
print("NEGM Results:")
print("-"*70)

for t in [0, 10, 19]:
    d = model.sol.d[t]
    print(f"\nt = {t}:")
    print(f"  d: min={d.min():.6f}, max={d.max():.6f}, mean={d.mean():.6f}")
    print(f"  Nonzero (>0.001): {np.sum(d > 0.001)}/{d.size}")
    print(f"  d > 0.1: {np.sum(d > 0.1)}")
    print(f"  d > 1.0: {np.sum(d > 1.0)}")

print("\n" + "="*70)

