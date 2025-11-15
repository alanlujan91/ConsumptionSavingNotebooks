# G2EGMSmooth - Taste Shock Smoothing for G2EGM

This implementation extends the standard G2EGM algorithm with taste shock smoothing over discrete choice alternatives, as described in the G2EGM literature.

## Key Features

- **Minimal code duplication**: All segment solution methods (`solve_ucon`, `solve_dcon`, `solve_acon`, `solve_con`) are imported directly from `G2EGM.py`
- **Single difference**: Only the aggregation step is modified - using smooth max (logsum) instead of discrete argmax
- **Backward compatible**: When `sigma=0`, behaves identically to standard G2EGM

## Implementation

The taste shock approach uses extreme value type I (Gumbel) distributed preference shocks with scale parameter `sigma`:

```
V(m,n) = E[max_j {v_j(m,n) + σ*ε_j}]
       = σ * log(Σ_j exp(v_j(m,n)/σ))  [logsum formula]
```

Where:
- `v_j` = value of segment j (ucon, dcon, acon, con)
- `σ` = taste shock scale parameter
- `ε_j` ~ EV Type I (Gumbel) with scale 1

Policy functions become probability-weighted averages:
```
c(m,n) = Σ_j P_j(m,n) * c_j(m,n)
d(m,n) = Σ_j P_j(m,n) * d_j(m,n)
```

Where choice probabilities follow the logit formula:
```
P_j(m,n) = exp(v_j(m,n)/σ) / Σ_k exp(v_k(m,n)/σ)
```

## Usage

In `G2EGMModel.py`:

```python
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

## Effect of `sigma`

- `σ = 0`: Standard discrete choice (equivalent to G2EGM)
- `σ → 0`: Approaches discrete choice
- `σ > 0`: Smooth choice with probability mixing
- Larger `σ`: More smoothing, more equal probabilities across alternatives

## Code Structure

**G2EGMSmooth.py** (this file):
- Imports all segment solvers from G2EGM
- Adds `logsum()` and `choice_prob()` helper functions
- Overrides only the `solve()` function's aggregation logic

**Difference from G2EGM.solve()**:
- Standard G2EGM: Uses `argmax` to pick best segment, then uses that segment's policy
- G2EGMSmooth: Uses `logsum` for value, probability-weighted average for policies

## Testing

See `test_smooth.py` for comparison tests and `compare_point.py` for detailed point-wise comparison.
