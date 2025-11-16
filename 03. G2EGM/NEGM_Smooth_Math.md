# NEGM Smooth - Mathematical Structure

## Model Overview

This is a **pension and retirement choice model** where households make three key decisions:
1. **Consumption** ($c$): How much to consume
2. **Pension contributions** ($d$): How much to contribute to pension wealth
3. **Work vs Retire**: Whether to continue working or retire (discrete choice with taste shocks)

State variables:
- $m$ = liquid wealth (cash-on-hand)
- $n$ = pension wealth (illiquid, accumulates from contributions)

The pension utility function $\psi(d)$ captures the benefit of pension contributions (e.g., tax advantages, employer matching).

## Step 0: Post-Decision Value with Taste Shocks (Work vs Retire)

### Alternatives

$$
\mathcal{J} = \{\text{work}, \text{retire}\}
$$

### Taste Shock Distribution

$$
\varepsilon_j \sim \text{EV Type I (Gumbel)}, \quad \forall j \in \mathcal{J}
$$

### Alternative Values (Next Period)

**Work alternative:**
$$
V_{\text{work},t+1}(m_{t+1}, n_{t+1}) = V_{t+1}(m_{t+1}, n_{t+1})
$$

**Retire alternative:**
$$
V_{\text{retire},t+1}(m_{t+1}, n_{t+1}) = V^{\text{ret}}_{t+1}(m_{t+1} + n_{t+1})
$$

where values are evaluated at post-decision states that evolve as:
- $m_{t+1} = R_a a^{\times} + y_{t+1}$ (liquid wealth next period: returns on liquid assets plus income)
- $n_{t+1} = R_b b^{\times}$ (pension wealth next period: returns on pension contributions)
- Upon retirement, pension wealth $n$ is liquidated and added to liquid wealth $m$

**Implementation note:** The algorithm works with inverse values $-1/V$ for numerical stability, but the mathematics below uses direct values for clarity.

### Smooth Max over Alternatives

**Standard (σ = 0):**
$$
\tilde{V}_{t+1}(m_{t+1}, n_{t+1}) = \max\{V_{\text{work},t+1}, V_{\text{retire},t+1}\}
$$

**Smooth (σ > 0):**
$$
\tilde{V}_{t+1}(m_{t+1}, n_{t+1}) = \mathbb{E}_{\varepsilon}\left[\max_{j \in \mathcal{J}} \{V_{j,t+1} + \sigma \varepsilon_j\}\right]
$$

Using properties of EV Type I distribution:
$$
= \sigma \log\left(\sum_{j \in \mathcal{J}} \exp\left(\frac{V_{j,t+1}}{\sigma}\right)\right)
$$

### Choice Probabilities (Logit Formula)

$$
P_{j,t+1}(m_{t+1}, n_{t+1}) = \frac{\exp(V_{j,t+1}/\sigma)}{\sum_{k \in \mathcal{J}} \exp(V_{k,t+1}/\sigma)}
$$

Specifically:
$$
P_{\text{work},t+1} = \frac{\exp(V_{\text{work},t+1}/\sigma)}{\exp(V_{\text{work},t+1}/\sigma) + \exp(V_{\text{retire},t+1}/\sigma)}
$$

$$
P_{\text{retire},t+1} = 1 - P_{\text{work},t+1}
$$

### Post-Decision Value

$$
w(b^{\times}, a^{\times}) = \beta \mathbb{E}_{y}\left[\tilde{V}_{t+1}(R_a a^{\times} + y_{t+1}, R_b b^{\times})\right]
$$

### Marginal Post-Decision Value

**Key insight:** By the envelope theorem for the smooth max operator:

$$
\frac{\partial \tilde{V}_{t+1}}{\partial m_{t+1}} = \sum_{j \in \mathcal{J}} P_{j,t+1} \cdot \frac{\partial V_{j,t+1}}{\partial m_{t+1}}
$$

Therefore:
$$
\frac{\partial \tilde{V}_{t+1}}{\partial m_{t+1}} = P_{\text{work},t+1} \cdot V_{m,\text{work},t+1} + P_{\text{retire},t+1} \cdot V^{\text{ret}}_{m,t+1}
$$

And the marginal post-decision value is:
$$
w_{a}(b^{\times}, a^{\times}) = \beta R_a \mathbb{E}_{y}\left[\frac{\partial \tilde{V}_{t+1}}{\partial m_{t+1}}\right]
$$

$$
= \beta R_a \mathbb{E}_{y}\left[P_{\text{work},t+1} \cdot V_{m,\text{work},t+1} + P_{\text{retire},t+1} \cdot V^{\text{ret}}_{m,t+1}\right]
$$

where the probabilities $P_{j,t+1}$ are evaluated at each realization of the income shock $y_{t+1}$, making them random variables inside the expectation.

**For G2EGM, similarly:**
$$
w_{b}(b^{\times}, a^{\times}) = \beta R_b \mathbb{E}_{y}\left[P_{\text{work},t+1} \cdot V_{n,\text{work},t+1} + P_{\text{retire},t+1} \cdot V^{\text{ret}}_{m,t+1}\right]
$$

Note: For retirement, $\frac{\partial V^{\text{ret}}}{\partial n} = \frac{\partial V^{\text{ret}}}{\partial m}$ since pension wealth $n$ is liquidated and added to liquid wealth $m$ upon retirement.

## Step 1: EGM - Solve Pure Consumption Problem

This step solves for optimal consumption $c$ conditional on pension contribution $d$ (equivalently, conditional on post-decision pension wealth $b^\times$).

### Post-Decision States

$$
b^{\times} = \text{post-decision pension wealth}
$$

$$
a^{\times} = \text{post-decision liquid assets}
$$

### State Evolution

$$
m_{t+1} = R_a a^{\times} + y_{t+1}
$$

$$
n_{t+1} = R_b b^{\times}
$$

### Post-Decision Value

$$
w(b^{\times}, a^{\times}) = \beta \mathbb{E}[V_{t+1}(R_a a^{\times} + y_{t+1}, R_b b^{\times})]
$$

### Marginal Post-Decision Value

$$
w_{a}(b^{\times}, a^{\times}) = \beta R_a \mathbb{E}[V_{m,t+1}(R_a a^{\times} + y_{t+1}, R_b b^{\times})]
$$

### Euler Equation

$$
u'(c) = w_{a}(b^{\times}, a^{\times})
$$

$$
c(b^{\times}, a^{\times}) = (u')^{-1}\left(w_{a}(b^{\times}, a^{\times})\right)
$$

### Endogenous Grid Method

$$
m^{\cap}(b^{\times}, a^{\times}) = a^{\times} + c(b^{\times}, a^{\times})
$$

### Upper Envelope and Regridding

**Inputs:** 
- Exogenous grid: $a^{\times}$
- Endogenous grid: $m^{\cap}(b^{\times}, a^{\times})$
- Functions on endogenous grid: $c(b^{\times}, a^{\times})$, $w(b^{\times}, a^{\times})$
- Target exogenous grid: $m^{\times}$

**Outputs:** 
- Functions on exogenous grid: $\mathbf{c}_{\text{pure}}(b^{\times}, m^{\times})$, $\mathbf{v}_{\text{pure}}(b^{\times}, m^{\times})$

$$
\left[\mathbf{c}_{\text{pure}}, \mathbf{v}_{\text{pure}}\right] = \text{upperenvelope}\left(a^{\times}, m^{\cap}, c, w \,|\, m^{\times}\right)
$$

## Step 2: VFI - Optimize Pension Contribution

This step optimizes the pension contribution $d$ given current state $(m, n)$.

### Current Period States

$$
n = \text{beginning-of-period pension wealth}
$$

$$
m = \text{liquid wealth (cash-on-hand)}
$$

### Post-Decision State Mapping

$$
b^{\times}(d) = n + d + \psi(d)
$$

$$
m^{\times}(d) = m - d
$$

where:
- $d$ = pension contribution (can be positive or negative, representing contributions or withdrawals)
- $\psi(d)$ = pension utility function (captures tax advantages, employer matching, etc.)
- $b^\times(d)$ = post-decision pension wealth (current pension $n$ plus contribution $d$ plus pension benefits $\psi(d)$)
- $m^\times(d)$ = remaining liquid wealth after pension contribution

### Objective Function (Negative Inverse Value)

The algorithm stores inverse values as $\text{inv\_v}_{\text{pure}} = -1/v_{\text{pure}}$.

The objective function for optimization is:
$$
\Phi(d; m, n) = -\text{inv\_v}_{\text{pure}}(b^{\times}(d), m^{\times}(d)) = \frac{1}{v_{\text{pure}}(b^{\times}(d), m^{\times}(d))}
$$

where:
$$
v_{\text{pure}}(b^{\times}, m^{\times}) = u(m^{\times}) + w(b^{\times}, 0)
$$

**Key insight:** Minimizing $\Phi = 1/v$ is equivalent to maximizing $v$ (since $v > 0$).

Therefore:
$$
d^*(m,n) = \arg\min_{d \in [0,m]} \Phi(d; m, n) = \arg\max_{d \in [0,m]} v_{\text{pure}}(b^{\times}(d), m^{\times}(d))
$$

### Interior Solution ($d^* \in (0, m)$)

$$
d^*_{\text{int}}(m, n) = \arg\min_{d \in (0, m)} \Phi(d; m, n)
$$

$$
b^{\times}_{\text{int}} = n + d^*_{\text{int}} + \psi(d^*_{\text{int}})
$$

$$
m^{\times}_{\text{int}} = m - d^*_{\text{int}}
$$

$$
c^*_{\text{int}} = \min\left\{\mathcal{I}^{2d}(b^{\times}_{\text{int}}, m^{\times}_{\text{int}}; \mathbf{c}_{\text{pure}}), m^{\times}_{\text{int}}\right\}
$$

$$
v_{\text{int}}(m, n) = -\Phi(d^*_{\text{int}}; m, n)
$$

### No Pension Contribution Corner ($d = 0$)

$$
b^{\times}_{\text{con}} = n
$$

$$
m^{\times}_{\text{con}} = m
$$

$$
c_{\text{dcon}}(m, n) = \mathcal{I}^{2d}(n, m; \mathbf{c}_{\text{pure}})
$$

$$
v_{\text{dcon}}(m, n) = -\Phi(0; m, n)
$$

### Corner Solution ($c = m, d = 0, a = 0$)

$$
c_{\text{con}}(m, n) = m
$$

$$
v_{\text{con}}(m, n) = u(m) + w(n, 0)
$$

## Step 3: Aggregate Solution

The value function $V(m, n)$ is determined by taking the maximum over the three regions:

$$
V(m, n) = \max\{v_{\text{int}}(m, n), v_{\text{dcon}}(m, n), v_{\text{con}}(m, n)\}
$$

The policy functions $(c, d)$ correspond to the region that achieves the maximum:
- **Interior:** $d^* > -n$ satisfies first-order condition
- **Pension-depleted:** $d^* = -n$ (withdraw all pension wealth)
- **No contribution:** $d^* = 0$ (maintain current pension wealth)

**Implementation note:** Since the algorithm stores $\text{inv\_v} = -1/v$, the comparison for the maximum is done in inverse space:
$$
\arg\max_k v_k = \arg\max_k (-1/v_k)
$$
because if $v_1 > v_2 > 0$, then $-1/v_1 > -1/v_2$ (both negative, but $-1/v_1$ is "less negative").

**Note:** These are not discrete choices with taste shocks - they are simply different regions of the pension contribution constraint set. The discrete choice with taste shocks occurs at the post-decision level (work vs retire) in Step 0.

## Step 4: Marginal Value Function

### Envelope Condition

$$
V_{m}(m, n) = u'(c(m, n))
$$

## Limiting Case: Discrete Retirement Choice (σ → 0)

### Post-Decision Value

$$
\lim_{\sigma \to 0} \tilde{V}_{t+1}(m_{t+1}, n_{t+1}) = \max\{V_{\text{work},t+1}, V_{\text{retire},t+1}\}
$$

### Choice Probabilities

$$
\lim_{\sigma \to 0} P_{j,t+1} = \begin{cases}
1 & \text{if } j = \arg\max_k V_{k,t+1} \\
0 & \text{otherwise}
\end{cases}
$$

### Marginal Post-Decision Value

In the limit, the probability-weighted derivative collapses to the discrete max:
$$
\lim_{\sigma \to 0} w_{a}(b^{\times}, a^{\times}) = \beta R_a \mathbb{E}_{y}\left[\max\left\{V_{m,\text{work},t+1}, V^{\text{ret}}_{m,t+1}\right\}\right]
$$

where the max is taken point-wise at each realization of $y_{t+1}$ inside the expectation.

This recovers the standard NEGM algorithm with discrete work/retire choice.

## Implementation Notes

### Working with Inverse Values

The code uses inverse values $\text{inv\_v} = -1/V$ for numerical stability. The key relationships are:

1. Higher inverse value → Lower actual value (because of the negative sign and division)
2. To get actual value: $V = -1/\text{inv\_v}$
3. For marginal values: $V_m = 1/\text{inv\_vm}$ (positive inverse)

### Numerical Stability in Logsumexp

The implementation uses the log-sum-exp trick:
$$
\log\left(\sum_j \exp(x_j)\right) = x_{\max} + \log\left(\sum_j \exp(x_j - x_{\max})\right)
$$

This prevents overflow/underflow when computing the smooth max.

### Monte Carlo Integration

The expectation $\mathbb{E}_y$ is computed via Gaussian quadrature over the income shock distribution:
$$
\mathbb{E}_y[f(y)] \approx \sum_{i=1}^{N_\eta} w_{\eta,i} \cdot f(y_i)
$$

where $\{y_i, w_{\eta,i}\}$ are the quadrature nodes and weights.
