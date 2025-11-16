# Taste Shock Smoothing for EGM Algorithms

This document describes the taste shock smoothing implementations for both G2EGM and NEGM algorithms. These implementations extend the standard discrete choice algorithms with smooth choice probabilities using extreme value type I (Gumbel) distributed preference shocks.

## Overview

Both `G2EGMSmooth.py` and `NEGMSmooth.py` implement the same fundamental smoothing approach:

- **Minimal code changes**: Import segment/alternative solvers from the standard implementations (`G2EGM.py` and `NEGM.py`)
- **Single modification**: Replace discrete choice (`argmax`) with smooth aggregation (`logsum`)
- **Backward compatible**: When `σ = 0`, both behave identically to their standard counterparts

## Mathematical Framework

### Taste Shocks

Both implementations use extreme value type I (Gumbel) distributed preference shocks with scale parameter `σ`:

```
V(states) = E[max_j {v_j(states) + σ*ε_j}]
          = σ * log(Σ_j exp(v_j(states)/σ))  [logsum formula]
```

Where:
- `v_j` = value of alternative j
- `σ` = taste shock scale parameter (controls degree of smoothing)
- `ε_j ~ EV Type I` (Gumbel) with scale 1

### Choice Probabilities

Under this framework, choice probabilities follow the multinomial logit formula:

```
P_j(states) = exp(v_j(states)/σ) / Σ_k exp(v_k(states)/σ)
```

### Policy Functions

Policy functions become probability-weighted averages across alternatives:

```
c(states) = Σ_j P_j(states) * c_j(states)
d(states) = Σ_j P_j(states) * d_j(states)
```

## G2EGMSmooth

### Alternatives

G2EGMSmooth aggregates over four discrete choice segments:
1. **ucon**: Unconstrained (optimal a > 0, d > 0)
2. **dcon**: Durables-constrained (optimal d = 0, a > 0)
3. **acon**: Assets-constrained (optimal a = 0, d > 0)
4. **con**: Corner solution (a = 0, d = 0)

### Implementation

**G2EGMSmooth.py**:
- Imports all segment solvers from `G2EGM.py`: `solve_ucon()`, `solve_dcon()`, `solve_acon()`, `solve_con()`
- Adds `logsum()` and `choice_prob()` helper functions
- Overrides only the `solve()` function's aggregation logic

**Difference from G2EGM.solve()**:
- Standard G2EGM: Uses `argmax` to pick best segment, then uses that segment's policy
- G2EGMSmooth: Uses `logsum` for value, probability-weighted average for policies

### Usage

```python
from G2EGMModel import G2EGMModelClass

model = G2EGMModelClass()

# Standard G2EGM (discrete choice)
model.par.solmethod = 'G2EGM'
model.par.sigma = 0.0  # not used

# Smooth G2EGM with taste shocks
model.par.solmethod = 'G2EGMSmooth'
model.par.sigma = 0.05  # small smoothing
# or
model.par.sigma = 0.1   # more smoothing

model.solve()
```

## NEGMSmooth

### Alternatives

NEGMSmooth aggregates over three discrete choice alternatives:
1. **Interior**: Optimal durables d* ∈ (0, m) with a* > 0
2. **dcon**: Durables-constrained (d = 0, a* > 0)
3. **Corner**: Corner solution (c = m, d = 0, a = 0)

### Implementation

**NEGMSmooth.py**:
- Imports pure consumption solver from `NEGM.py`: `solve_pure_c()`, `obj_outer()`
- Adds `logsum()` and `choice_prob()` helper functions
- Modifies only the `solve_outer()` function's aggregation logic

**Difference from NEGM.solve_outer()**:
- Standard NEGM: Uses discrete comparisons to pick best alternative
- NEGMSmooth: Uses `logsum` for value, probability-weighted average for policies
- Handles invalid alternatives (NaN, inf) by filtering before aggregation

### Usage

```python
from G2EGMModel import G2EGMModelClass

model = G2EGMModelClass()

# Standard NEGM (discrete choice)
model.par.solmethod = 'NEGM'
model.par.sigma = 0.0  # not used

# Smooth NEGM with taste shocks
model.par.solmethod = 'NEGMSmooth'
model.par.sigma = 0.05  # small smoothing
# or
model.par.sigma = 0.1   # more smoothing

model.solve()
```

## Effect of σ Parameter

For both implementations:

- **σ = 0**: Standard discrete choice (equivalent to G2EGM/NEGM)
- **σ → 0**: Approaches discrete choice as σ decreases
- **σ > 0**: Smooth choice with probability mixing
  - Small σ (e.g., 0.01-0.05): Minimal smoothing, nearly discrete
  - Medium σ (e.g., 0.05-0.1): Moderate smoothing
  - Large σ (e.g., > 0.1): Heavy smoothing, more equal probabilities

**Trade-offs**:
- Larger σ: Smoother value/policy functions, better numerical stability
- Smaller σ: Closer to optimal discrete choice, sharper transitions

## Testing

Both implementations include comprehensive test suites:

**test_G2EGMSmooth.py**:
- Basic functionality tests
- Comparison with G2EGM (σ = 0 should match)
- Smoothness verification
- Timing comparisons

**test_NEGMSmooth.py**:
- Basic functionality tests
- Comparison with NEGM (σ = 0 should match)
- Smoothness verification
- Timing comparisons

Run tests with:
```python
# Test G2EGMSmooth
from test_G2EGMSmooth import run_all_tests as test_g2egm
test_g2egm()

# Test NEGMSmooth
from test_NEGMSmooth import run_all_tests as test_negm
test_negm()
```

## Code Structure

Both implementations follow the same design pattern:

1. **Import base algorithm components** - Leverage existing segment/alternative solvers
2. **Add smoothing utilities** - `logsum()` and `choice_prob()` functions
3. **Modify aggregation only** - Replace discrete max with smooth max
4. **Maintain compatibility** - σ = 0 recovers standard algorithm

This minimal-change approach ensures:
- Easy maintenance (changes to base algorithms propagate automatically)
- Clear understanding of differences
- Confidence in correctness (only aggregation differs)

## References

The taste shock smoothing approach is widely used in structural models to:
- Improve numerical stability
- Generate smooth value and policy functions
- Facilitate estimation (smooth likelihood functions)
- Handle near-indifference between alternatives

For G2EGM, see the original papers on the generalized endogenous grid method with discrete-continuous choice.
