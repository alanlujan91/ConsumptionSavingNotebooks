# NEGM Smooth - Mathematical Structure

# NEGM Smooth - Mathematical Structure

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

$$
V_{\text{work},t+1}(m_{t+1}, n_{t+1}) = V_{t+1}(m_{t+1}, n_{t+1})
$$

$$
V_{\text{retire},t+1}(m_{t+1}, n_{t+1}) = V^{\text{ret}}_{t+1}(m_{t+1} + n_{t+1})
$$

where values are evaluated at $(m_{t+1}, n_{t+1}) = (R_a a^{\times} + y_{t+1}, R_b b^{\times})$.

### Smooth Max over Alternatives

**Standard (σ = 0):**
$$
\tilde{V}_{t+1}(m_{t+1}, n_{t+1}) = \max\{V_{\text{work},t+1}, V_{\text{retire},t+1}\}
$$

**Smooth (σ > 0):**
$$
\tilde{V}_{t+1}(m_{t+1}, n_{t+1}) = \mathbb{E}_{\varepsilon}\left[\max_{j \in \mathcal{J}} \{V_{j,t+1} + \sigma \varepsilon_j\}\right]
$$

$$
= \sigma \log\left(\sum_{j \in \mathcal{J}} \exp\left(\frac{V_{j,t+1}}{\sigma}\right)\right)
$$

### Post-Decision Value

$$
w(b^{\times}, a^{\times}) = \beta \mathbb{E}_{y}\left[\tilde{V}_{t+1}(R_a a^{\times} + y_{t+1}, R_b b^{\times})\right]
$$

### Marginal Post-Decision Value

For smooth case (σ > 0), marginal value is probability-weighted:

$$
P_{\text{work},t+1} = \frac{\exp(V_{\text{work},t+1}/\sigma)}{\exp(V_{\text{work},t+1}/\sigma) + \exp(V_{\text{retire},t+1}/\sigma)}
$$

$$
P_{\text{retire},t+1} = \frac{\exp(V_{\text{retire},t+1}/\sigma)}{\exp(V_{\text{work},t+1}/\sigma) + \exp(V_{\text{retire},t+1}/\sigma)}
$$

$$
w_{a}(b^{\times}, a^{\times}) = \beta R_a \mathbb{E}_{y}\left[P_{\text{work},t+1} \cdot V_{m,t+1} + P_{\text{retire},t+1} \cdot V^{\text{ret}}_{m,t+1}\right]
$$

## Step 1: EGM - Solve Pure Consumption Problem

### Post-Decision States

$$
b^{\times} = \text{post-decision durables}
$$

$$
a^{\times} = \text{post-decision assets}
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

## Step 2: VFI - Optimize Durables Choice

### Current Period States

$$
n = \text{beginning-of-period durables}
$$

$$
m = \text{cash-on-hand}
$$

### Post-Decision State Mapping

$$
b^{\times}(d) = n + d + \psi(d)
$$

$$
m^{\times}(d) = m - d
$$

### Objective Function (Negative Inverse Value)

$$
\Phi(d; m, n) = -\frac{1}{u(m^{\times}(d)) + w(b^{\times}(d), 0)}
$$

Note: Minimize $\Phi$ is equivalent to maximizing value since $v = -1/\Phi$.

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

### Durables Constrained ($d = 0$)

$$
b^{\times}_{\text{dcon}} = n
$$

$$
m^{\times}_{\text{dcon}} = m
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

The value function $V(m,n)$ is determined by taking the maximum over the three regions:

$$
V(m, n) = \max\{v_{\text{int}}(m, n), v_{\text{dcon}}(m, n), v_{\text{con}}(m, n)\}
$$

The policy functions $(c, d)$ correspond to the region that achieves the maximum.

**Note:** These are not discrete choices with taste shocks - they are simply different regions of the constraint set. The discrete choice with taste shocks occurs at the post-decision level (work vs retire) in Step 0.

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

This recovers the standard NEGM algorithm with discrete work/retire choice.
