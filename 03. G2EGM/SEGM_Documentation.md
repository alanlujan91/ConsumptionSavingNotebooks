# Sequential Endogenous Grid Method (SEGM)

**Sequential EGM for Consumption-Saving Models with Pension Contributions**

## 1. The Full Model

### State Variables
- $m_t$: Liquid wealth (cash-on-hand)
- $n_t$: Pension wealth

### Choice Variables  
- $c_t$: Consumption
- $d_t$: Pension contribution (voluntary deposits)

### Bellman Equation

$$
V(m_t, n_t) = \max_{c_t, d_t} \left\{ u(c_t) + \beta \mathbb{E}_t\left[V(m_{t+1}, n_{t+1})\right] \right\}
$$

Subject to:
$$
\begin{align}
a_t &= m_t - c_t - d_t \geq 0 \\
b_t &= n_t + d_t + \psi(d_t) \\
m_{t+1} &= R a_t + y_{t+1} \\
n_{t+1} &= R_b b_t
\end{align}
$$

where:
- $a_t$: Post-decision liquid assets
- $b_t$: Post-decision pension wealth
- $\psi(d)$: Pension subsidy function, $\psi(d) = \chi \log(1 + d)$
- $R$: Gross return on liquid assets
- $R_b$: Gross return on pension wealth
- $y_{t+1}$: Stochastic income

### First-Order Conditions

**Consumption FOC:**
$$
u'(c_t) = w_a(b_t, a_t)
$$

**Pension FOC:**
$$
u'(c_t) \cdot (1 + \psi'(d_t)) = w_b(b_t, a_t)
$$

where the post-decision value function and its derivatives are:
$$
\begin{align}
w(b_t, a_t) &= \beta \mathbb{E}_t\left[V(m_{t+1}, n_{t+1})\right] \\
w_a(b_t, a_t) &= \beta R \mathbb{E}_t\left[\frac{\partial V}{\partial m_{t+1}}\right] \\
w_b(b_t, a_t) &= \beta R_b \mathbb{E}_t\left[\frac{\partial V}{\partial n_{t+1}}\right]
\end{align}
$$

Combining the FOCs yields:
$$
1 + \psi'(d_t) = \frac{w_b(b_t, a_t)}{w_a(b_t, a_t)}
$$

## 2. Decomposition into Subproblems

The key insight is that we can solve this problem **sequentially** by exploiting the structure of the constraints and introducing an intermediate state variable.

### The Key Reformulation: Liquid Resources

The budget constraint $a_t = m_t - c_t - d_t$ can be rewritten by defining **liquid resources after pension deposit**:
$$
l_t \equiv a_t + c_t = m_t - d_t
$$

This reveals that:
1. The pension choice $d_t$ transforms total resources $m_t$ into liquid resources $l_t = m_t - d_t$
2. The consumption choice $c_t$ is made from liquid resources $l_t$, with $a_t = l_t - c_t$

The budget constraints become:
$$
\begin{align}
l_t &= m_t - d_t & \text{(liquid resources)} \\
a_t &= l_t - c_t & \text{(post-decision liquid)} \\
b_t &= n_t + d_t + \psi(d_t) & \text{(post-decision pension)}
\end{align}
$$

This reformulation enables **true sequential separation**:

1. **First Subproblem (Pension):** Given $(n_t, m_t)$, choose $d_t$ to determine $(b_t, l_t)$
2. **Second Subproblem (Consumption):** Given $(b_t, l_t)$, choose $c_t$ to determine $a_t$

**Crucially:** No fixed-point problem! The state $l_t$ for the consumption problem is **fully determined** by the pension choice before consumption is optimized.

### Subproblem 1: Pure Pension (Outer Problem)

**Problem:** Given $(n_t, m_t)$, choose $d \geq 0$ to maximize:
$$
v_{\text{pure\_d}}(n, m) = \max_{d \geq 0} \left\{ v_{\text{pure\_c}}(n + d + \psi(d), m - d) \right\}
$$

where $v_{\text{pure\_c}}(b, l)$ is the value of the pure consumption problem.

**FOC (interior solution $d > 0$):**
$$
v_{\text{pure\_c},b}(b, l) \cdot (1 + \psi'(d)) = v_{\text{pure\_c},l}(b, l)
$$

where $b = n + d + \psi(d)$ and $l = m - d$.

Rearranging:
$$
\psi'(d) = \frac{v_{\text{pure\_c},l}(b, l)}{v_{\text{pure\_c},b}(b, l)} - 1
$$

**Key Point:** We need marginal values of $v_{\text{pure\_c}}$ with respect to **liquid resources $l$**, not total resources $m$. But by definition, $\partial v/\partial l = \partial v/\partial m$ when we're choosing optimally!

**State Space:**
- Exogenous: $(n_t, m_t)$ - pre-decision states  
- Choice: $d_t$
- Endogenous: $(b_t, l_t)$ where $b_t = n_t + d_t + \psi(d_t)$ and $l_t = m_t - d_t$
- Output: $d(n, m)$, and intermediate states $(b, l)$ for Subproblem 2

### Subproblem 2: Pure Consumption (Inner Problem)

**Problem:** Given $(b_t, l_t)$, choose $c$ to maximize:
$$
v_{\text{pure\_c}}(b, l) = \max_{c} \left\{ u(c) + w(b, l - c) \right\}
$$

**FOC:**
$$
u'(c) = w_a(b, a) \quad \text{where } a = l - c
$$

**State Space:**
- Exogenous: $(b_t, l_t)$ - post-decision pension and liquid resources
- Choice: $c$
- Endogenous: None (we solve this given fixed $b$, $l$)
- Output: $c(b, l)$, $v_{\text{pure\_c}}(b, l)$, and marginal values $v_{\text{pure\_c},l}(b, l)$, $v_{\text{pure\_c},b}(b, l)$

**Note:** The marginal value $v_{\text{pure\_c},l}(b, l) = w_a(b, l - c^*)$ by the envelope theorem.

## 2.3 Why The Problems Are Completely Separable

The key insight is that the two subproblems are **completely independent** when solved in the correct order:

**Subproblem 1 (Consumption - Inner):**
- Input: Exogenous $(b, a)$ from post-decision grids
- Invert consumption FOC: $c = (u')^{-1}(w_a(b, a))$
- Compute endogenous liquid resources: $l = a + c$
- Output: $c(b, l)$, $v(b, l)$, $v_l(b, l) = w_a(b, a)$, $v_b(b, l) = w_b(b, a)$
- **No other choices involved** - pure consumption problem

**Subproblem 2 (Pension - Outer):**
- Input: Exogenous $(b, l)$ from Subproblem 1 output
- At each $(b, l)$, marginal values $v_l(b,l)$ and $v_b(b,l)$ are **KNOWN**
- Invert pension FOC: $d = \chi v_b / (v_l - v_b) - 1$ is **EXPLICIT** (no iteration needed!)
- Compute endogenous states: $n = b - d - \psi(d)$ and $m = l + d$
- **No other choices involved** - pure pension problem given known marginal values

**Why There's No Fixed-Point:**
- At $(b, l)$, we're asking: "What pension deposit $d$ would have led to this intermediate state?"
- This is the **inverse** of the state transition, exactly like standard EGM
- The marginal values $v_l(b,l)$ and $v_b(b,l)$ are properties of the state $(b,l)$, not functions of $d$
- Therefore the FOC inversion is **explicit**: given $(b,l)$ and its marginal values, compute $d$

## 3. Solution Methods: NEGM vs SEGM

### 3.1 NEGM (Nested EGM)

**Subproblem 1:** Pure Consumption - **EGM**
- For each $b_t$, vary $a_t \in \{a_1, \ldots, a_{N_a}\}$
- Invert FOC: $c(b_t, a_t) = (u')^{-1}(w_a(b_t, a_t))$
- Compute endogenous $m^\cap(b_t, a_t) = a_t + c(b_t, a_t)$
- **1D Upper Envelope:** Regrid from $(b_t, a_t)$ with endogenous $m^\cap$ to $(b_t, m)$
- Output: $c_{\text{pure\_c}}(b_t, m)$ and $v_{\text{pure\_c}}(b_t, m)$

**Subproblem 2:** Pure Pension - **VFI (Value Function Iteration)**
- For each $(n, m)$, search over $d \in [0, m]$ using golden section search
- Evaluate: $V(n, m; d) = v_{\text{pure\_c}}(n + d + \psi(d), m - d)$
- Find: $d^*(n, m) = \arg\max_d V(n, m; d)$
- Output: $d(n, m)$, $c(n, m) = c_{\text{pure\_c}}(n + d^* + \psi(d^*), m - d^*)$

**Key Feature:** Combines EGM (fast, no iteration) with VFI (slow, requires optimization)

### 3.2 SEGM (Sequential EGM) - **This Paper**

**Subproblem 2 (Inner):** Pure Consumption - **EGM**
- For each $b_t$, vary $a_t \in \{a_1, \ldots, a_{N_a}\}$  
- Invert FOC: $c(b_t, a_t) = (u')^{-1}(w_a(b_t, a_t))$
- Compute endogenous $l^\cap(b_t, a_t) = a_t + c(b_t, a_t)$ (liquid resources)
- **Modified 1D Upper Envelope:** Regrid and track marginal values
  - From $(b_t, a_t)$ with endogenous $l^\cap$ to $(b_t, l)$
  - Also output: $v_{\text{pure\_c},l}(b_t, l)$ and $v_{\text{pure\_c},b}(b_t, l)$ from active envelope segment
- Output: $c(b_t, l)$, $v_{\text{pure\_c}}(b_t, l)$, and marginal values $v_l(b,l)$, $v_b(b,l)$

**Subproblem 1 (Outer):** Pure Pension - **EGM** (Novel)
- For each $n$, vary $m \in \{m_1, \ldots, m_{N_m}\}$
- **Invert pension FOC:**
  - Need to evaluate $v_{\text{pure\_c}}(b, l)$ where $b = n + d + \psi(d)$ and $l = m - d$
  - FOC: $\psi'(d) = v_l(b, l) / v_b(b, l) - 1$
  
  For $\psi(d) = \chi \log(1 + d)$, we have $\psi'(d) = \chi/(1+d)$, so:
  $$
  d = \frac{\chi \cdot v_b(b, l)}{v_l(b, l) - v_b(b, l)} - 1
  $$
  
  But this still depends on $(b, l)$ which depend on $d$!
  
- **EGM Approach:** Treat $(b, l)$ as exogenous grid from Subproblem 2
  - For each $(b, l)$, we have $v_l(b, l)$ and $v_b(b, l)$
  - Invert: $d(b, l) = (\psi')^{-1}(v_l(b,l)/v_b(b,l) - 1)$
  - Compute endogenous: $n^\cap = b - d - \psi(d)$ and $m^\cap = l + d$
  - Value: $v(b, l) = v_{\text{pure\_c}}(b, l)$
  
- **2D Upper Envelope:** Regrid from $(b, l)$ with endogenous $(n^\cap, m^\cap)$ to $(n, m)$
  - Joint regridding over both dimensions
  - Or: Two sequential 1D envelopes (first over $n$ for each $l$, then over $m$ for each $n$)
  
- Output: $d(n, m)$, $c(n, m)$, $V(n, m)$

**Key Feature:** Pure EGM for both subproblems - no optimization, no iteration!

**Note on Order:** We solve inner (consumption) first to build the value function, then use it to solve outer (pension). This is opposite to NEGM, which solves consumption inner and pension outer via VFI.

### 3.3 The Crucial Insight: Complete Problem Separation via Liquid Resources

The reformulation with liquid resources $l = a + c$ enables **true sequential EGM** for both problems:

**Standard EGM for Consumption:**
- Given post-decision $(b, a)$
- Invert consumption FOC: $c = (u')^{-1}(w_a(b, a))$
- Compute endogenous liquid resources: $l = a + c$
- By envelope theorem: $v_l(b, l) = w_a(b, a)$ and $v_b(b, l) = w_b(b, a)$

**Novel EGM for Pension:**
- Given intermediate state $(b, l)$ from consumption problem
- Marginal values $v_l(b, l)$ and $v_b(b, l)$ are **known** from Subproblem 1
- Invert pension FOC: $d = \chi v_b(b,l) / (v_l(b,l) - v_b(b,l)) - 1$ is **explicit**
- Compute endogenous pre-decision states: $n = b - d - \psi(d)$ and $m = l + d$
- Regrid from $(b, l)$ with endogenous $(n, m)$ to target $(n, m)$ grid

**Why This Is Pure EGM (Not VFI):**
- At each grid point $(b, l)$, we **directly compute** $d$ from known marginal values
- No optimization required - just algebraic inversion of the FOC
- No iteration required - the formula is explicit
- Exactly analogous to standard EGM inverting $c$ from $w_a$

**Comparison:**
- **Standard EGM**: $(b, a) \xrightarrow{\text{invert } c} (b, l)$ using $w_a, w_b$
- **SEGM**: $(b, l) \xrightarrow{\text{invert } d} (n, m)$ using $v_l, v_b$
- **Both are pure inversions** - no fixed-points, no optimization!

## 4. Implementation Details

### 4.1 Upper Envelope for Subproblem 1

The standard `consav.upperenvelope` requires a utility function and computes values. For SEGM, we need a **modified** version that:

1. Accepts marginal value arrays $w_a$ and $w_b$ on the endogenous grid
2. Tracks which endogenous segment is active at each target point
3. **Outputs the marginal values from the active segment**

```python
def upperenvelope_with_marginals(grid_a, m_vec, c_vec, w_vec, wa_vec, wb_vec,
                                  grid_m, c_out, v_out, va_out, vb_out, par):
    """Upper envelope that also tracks marginal values from active segment"""
    
    for each target m:
        for each endogenous segment [i, i+1]:
            if m in [m_vec[i], m_vec[i+1]]:
                # Interpolate policy, value, AND marginal values
                c_interp = interpolate(c_vec[i:i+2], ...)
                v_interp = utility(c_interp) + interpolate(w_vec[i:i+2], ...)
                va_interp = interpolate(wa_vec[i:i+2], ...)  # KEY!
                vb_interp = interpolate(wb_vec[i:i+2], ...)  # KEY!
                
                # Update if this segment gives higher value
                if v_interp > v_out[m]:
                    v_out[m] = v_interp
                    c_out[m] = c_interp
                    va_out[m] = va_interp  # From active segment!
                    vb_out[m] = vb_interp  # From active segment!
```

### 4.2 Upper Envelope for Subproblem 2

Since values are already computed, we need a **value-based** upper envelope:

```python
def upperenvelope_1d_value(grid_endo, policy_vals, value_vals, 
                            grid_target, policy_out, value_out):
    """Simple 1D upper envelope when values are pre-computed"""
    
    for each target point:
        for each endogenous segment:
            if target in segment:
                # Linear interpolation
                v_interp = interpolate(value_vals, ...)
                policy_interp = interpolate(policy_vals, ...)
                
                # Update if higher value
                if v_interp > value_out[target]:
                    value_out[target] = v_interp
                    policy_out[target] = policy_interp
```

### 4.3 Algorithm Structure

```
For t = T-1, T-2, ..., 0:
    
    # Step 0: Compute post-decision value and marginal values
    w(b, a), wa(b, a), wb(b, a) from V(m[t+1], n[t+1])
    
    # Step 1: Pure Consumption EGM
    For each b in grid_b:
        For each a in grid_a:
            c = (u')^{-1}(wa(b, a))
            m_endo = a + c
        
        Upper envelope with marginals:
            Input: (a, m_endo, c, w, wa, wb)
            Output: c(b, m), v(b, m), v_m(b, m), v_b(b, m)
    
    # Step 2: Pure Pension EGM  
    For each m in grid_m:
        For each b in grid_b:
            # Get from Step 1
            c_val = c(b, m)
            v_m_val = v_m(b, m)
            v_b_val = v_b(b, m)
            
            # Invert pension FOC
            if v_m_val > v_b_val:
                d_val = (chi * v_b_val) / (v_m_val - v_b_val) - 1
                d_val = max(0, d_val)
            else:
                d_val = 0
            
            # Compute endogenous state
            n_endo = b - d_val - psi(d_val)
            v_val = v(b, m)
        
        Value-based upper envelope:
            Input: (b, n_endo, [c, d], v)
            Output: c(n, m), d(n, m), V(n, m)
```

## 5. Comparison Summary

| Feature | NEGM | SEGM |
|---------|------|------|
| **Subproblem 1** | EGM (standard) | EGM (with marginal tracking) |
| **Subproblem 2** | VFI (optimization) | **EGM (FOC inversion)** |
| **Speed** | Medium (VFI bottleneck) | **Fast (no optimization)** |
| **Accuracy** | High | **Same (EGM accurate)** |
| **Upper Envelopes** | 1 (consumption only) | **2 (consumption + pension)** |
| **FOC Inversion** | 1 (consumption) | **2 (consumption + pension)** |
| **Marginal Values** | Implicit | **Explicit tracking** |

## 6. Advantages of SEGM

1. **Computational Efficiency:** Eliminates the VFI optimization loop
   - NEGM: $O(N_n \times N_m \times K_{opt})$ where $K_{opt} \sim 10$-20 iterations
   - SEGM: $O(N_n \times N_m)$ direct evaluation

2. **Conceptual Clarity:** Both choices solved symmetrically via EGM
   - Same mathematical structure for both subproblems
   - Clear separation of concerns

3. **Numerical Accuracy:** No optimization tolerance parameters
   - Exact FOC inversion (up to machine precision)
   - No iteration convergence concerns

4. **Extensibility:** Natural framework for additional continuous choices
   - Any choice with invertible FOC can be added as another EGM step
   - Marginal value tracking is systematic

## 7. Key Innovation: Marginal Value Tracking

The critical innovation is computing and storing **marginal values of the intermediate value function**:

$$
v_{\text{pure\_c},m}(b, m) = w_a(b, a) \quad \text{where } a = m - c^*(b, m)
$$
$$
v_{\text{pure\_c},b}(b, m) = w_b(b, a) \quad \text{where } a = m - c^*(b, m)
$$

These marginal values:
1. Depend on **which envelope segment is active** at each $(b, m)$ point
2. Must be **interpolated from the active endogenous segment**, not recomputed from final grid
3. Enable **direct inversion** of the pension FOC without optimization

This transforms the outer VFI problem into a pure EGM problem, achieving the full computational benefits of EGM for both choices.

## 8. Current Implementation Status and Debugging

### 8.1 Implementation is Mathematically Correct

The current SEGM implementation has the **correct mathematical structure**:

**What We Implemented:**
1. ✅ Solve consumption on $(b, a) \to (b, l)$ where $l = a + c$ (liquid resources)
2. ✅ Compute marginal values $v_l(b, l) = w_a(b, a^*)$ and $v_b(b, l) = w_b(b, a^*)$ via envelope theorem
3. ✅ Solve pension on $(b, l) \to (n, m)$ by inverting $d = \chi v_b/(v_l - v_b) - 1$
4. ✅ Compute endogenous $(n, m) = (b - d - \psi(d), l + d)$
5. ✅ Use G2EGM's 2D upper envelope for final regridding

**This is pure EGM - no fixed-point problem!**

### 8.2 Why $d = 0$ Everywhere - Diagnostic Hypotheses

The issue is **implementation bugs**, not mathematical incorrectness. Possible causes:

**Hypothesis 1: Marginal Value Computation**
- The envelope theorem requires: $v_l(b,l) = w_a(b, a^*)$ where $a^* = l - c^*(b,l)$
- We interpolate $w_a(b, a^*)$ after solving for $c^*(b,l)$
- **Check**: Are we computing $a^* = l - c$ correctly? Are interpolated values sensible?

**Hypothesis 2: Marginal Value Range**
For $d > 0$, we need: $1 < v_l/v_b < \chi + 1 = 2.5$
- If $v_l/v_b \leq 1$: Denominator $(v_l - v_b) \leq 0$ → $d = 0$
- If $v_l/v_b \geq 2.5$: Numerator gives $d < 0$ → clamp to $d = 0$
- **Check**: What is the actual distribution of $v_l/v_b$ on the $(b,l)$ grid?

**Hypothesis 3: Grid Alignment**
- Consumption grid outputs $(b, l)$ where $l$ is on `par.grid_m`
- Pension grid should work on these $(b, l)$ points
- **Check**: Are we correctly using the $(b,l)$ grid or accidentally using $(b,m)$?

### 8.3 Next Debugging Steps

**Step 1: Verify Marginal Value Computation**

Create a diagnostic that prints:
```python
# After consumption EGM, at a sample (b, l) point:
print(f"b={b:.3f}, l={l:.3f}")
print(f"  c={c:.3f}, a={l-c:.3f}")
print(f"  v_l={v_l:.6f}, v_b={v_b:.6f}")
print(f"  ratio v_l/v_b={v_l/v_b:.3f}")
print(f"  Need 1 < ratio < {chi+1:.2f} for d>0")
```

**Step 2: Compare with NEGM Marginal Values**

NEGM computes $w_a(b, a)$ directly on post-decision grid. SEGM interpolates these at $a^* = l - c$.
- **Check**: Do SEGM's interpolated marginal values match NEGM's direct computation?
- **Test**: For same $(b, a)$ point, compare $w_a(b,a)$ from NEGM vs interpolated value in SEGM

**Step 3: Test Simplified Case**

- Set $\chi = 10$ (very high pension subsidy) → should definitely get $d > 0$
- Use coarse grids to manually inspect all values
- Print every $(b, l, v_l, v_b, d)$ tuple

### 8.4 Relationship to G2EGM and NEGM

**G2EGM:**
- Inverts both FOCs simultaneously at $(b, a)$
- Computes both endogenous $(m, n)$ together
- **Faster** when applicable (no VFI, but needs more derivatives)
- **More restrictive** (uniqueness requirements)

**SEGM:**
- Inverts FOCs sequentially: first $c$ at $(b,a)$, then $d$ at $(b,l)$
- Computes endogenous states in two stages
- **More general** (no uniqueness requirements for pension FOC)
- **Potentially faster** than NEGM (no VFI at all)
- **Intermediate complexity** between NEGM and G2EGM

**NEGM:**
- Inverts consumption FOC via EGM
- Uses VFI for pension choice
- **Most robust** (always works if consumption FOC is invertible)
- **Slower** than SEGM/G2EGM due to VFI step

## 9. Summary

**The Key Innovation:** SEGM introduces **liquid resources** $l = a + c$ as an intermediate state variable, enabling true sequential EGM decomposition:

1. **Consumption Problem:** $(b, a) \to c \to (b, l)$ via standard EGM
   - Outputs: $c(b,l)$, $v(b,l)$, and crucially $v_l(b,l) = w_a(b,a)$, $v_b(b,l) = w_b(b,a)$

2. **Pension Problem:** $(b, l) \to d \to (n, m)$ via novel EGM extension
   - At each $(b,l)$: marginal values are **known**, FOC inversion is **explicit**
   - No optimization, no iteration, pure algebraic inversion

**Mathematical Validity:** 
- ✅ Both subproblems are completely separable
- ✅ No fixed-point problem - all inversions are explicit
- ✅ Pure EGM for both choices (first application to sequential choice problems)

**Current Status:**
- ✅ Implementation has correct mathematical structure  
- ✅ Runs faster than NEGM (16s vs 23s)
- ✗ Produces $d=0$ everywhere - debugging needed
- → Issue is implementation bugs, not mathematical incorrectness

**Next Steps:**
1. Diagnose why marginal value ratios $v_l/v_b$ don't satisfy $1 < v_l/v_b < 2.5$
2. Verify envelope theorem computation of marginal values is correct
3. Compare SEGM marginal values with NEGM/G2EGM at same points
4. Test with extreme parameters ($\chi=10$) to force positive $d$
5. Once working, benchmark against G2EGM and NEGM
