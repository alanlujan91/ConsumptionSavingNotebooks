"""Test if G2EGM and NEGM solvers are working at all"""

import numpy as np
from G2EGMModel import G2EGMModelClass

print("Testing G2EGM...")
model_g2egm = G2EGMModelClass(name='G2EGM')
model_g2egm.par.solmethod = 'G2EGM'
model_g2egm.par.T = 20
try:
    model_g2egm.solve()
    print(f"✓ G2EGM solved")
    print(f"  d at t=0: mean={model_g2egm.sol.d[0].mean():.3f}, max={model_g2egm.sol.d[0].max():.3f}")
    print(f"  c at t=0: mean={model_g2egm.sol.c[0].mean():.3f}, max={model_g2egm.sol.c[0].max():.3f}")
except Exception as e:
    print(f"✗ G2EGM failed: {e}")

print("\nTesting NEGM...")
model_negm = G2EGMModelClass(name='NEGM')
model_negm.par.solmethod = 'NEGM'
model_negm.par.T = 20
try:
    model_negm.solve()
    print(f"✓ NEGM solved")
    print(f"  d at t=0: mean={model_negm.sol.d[0].mean():.3f}, max={model_negm.sol.d[0].max():.3f}")
    print(f"  c at t=0: mean={model_negm.sol.c[0].mean():.3f}, max={model_negm.sol.c[0].max():.3f}")
except Exception as e:
    print(f"✗ NEGM failed: {e}")

print("\nDone")

