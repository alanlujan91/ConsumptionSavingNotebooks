import numpy as np
from numba import njit

# consav
from consav import linear_interp
from consav import upperenvelope

# local modules
import utility
import pens
import NEGM
import upperenvelope as ue_local  # G2EGM's 2D upper envelope

# Create upper envelope for consumption subproblem (same as NEGM)
segm_upperenvelope = upperenvelope.create(utility.func, use_inv_w=False)

@njit
def upperenvelope_with_marginals(grid_a, m_vec, c_vec, w_vec, wa_vec, wb_vec,
                                  grid_m, c_out, v_out, va_out, vb_out, par):
    """Upper envelope that also tracks marginal values from the active segment"""
    
    Na = grid_a.size
    Nm = grid_m.size
    
    c_out[:] = 0.0
    v_out[:] = -np.inf
    va_out[:] = 0.0
    vb_out[:] = 0.0
    
    # Constraint: consume all if m < m_vec[0]
    im = 0
    while im < Nm and grid_m[im] <= m_vec[0]:
        c_out[im] = grid_m[im]
        v_out[im] = utility.func(c_out[im], par) + w_vec[0]
        va_out[im] = wa_vec[0]  # Marginal value from constraint point
        vb_out[im] = wb_vec[0]
        im += 1
    
    # Upper envelope over interior segments
    for ia in range(Na - 1):
        a_low = grid_a[ia]
        a_high = grid_a[ia + 1]
        
        if a_low > a_high:
            continue
        
        m_low = m_vec[ia]
        m_high = m_vec[ia + 1]
        c_low = c_vec[ia]
        c_high = c_vec[ia + 1]
        
        # Skip if degenerate segment
        if abs(m_high - m_low) < 1e-12:
            continue
        
        c_slope = (c_high - c_low) / (m_high - m_low)
        
        # Marginal values at segment endpoints
        wa_low, wa_high = wa_vec[ia], wa_vec[ia + 1]
        wb_low, wb_high = wb_vec[ia], wb_vec[ia + 1]
        wa_slope = (wa_high - wa_low) / (m_high - m_low)
        wb_slope = (wb_high - wb_low) / (m_high - m_low)
        
        for im in range(Nm):
            m = grid_m[im]
            
            if m >= m_low and m <= m_high:
                # Interpolate consumption
                c = c_low + c_slope * (m - m_low)
                
                # Interpolate post-decision asset
                a = a_low + (m - m_low) * (a_high - a_low) / (m_high - m_low)
                
                # Value
                v = utility.func(c, par) + (w_vec[ia] + (m - m_low) * 
                    (w_vec[ia + 1] - w_vec[ia]) / (m_high - m_low))
                
                # Update if this segment gives higher value
                if v > v_out[im]:
                    v_out[im] = v
                    c_out[im] = c
                    # Marginal values from this segment
                    va_out[im] = wa_low + wa_slope * (m - m_low)
                    vb_out[im] = wb_low + wb_slope * (m - m_low)

@njit
def solve_pure_c(t, sol, par):
    """Pure consumption EGM - outputs on (b, l) grid where l is LIQUID resources
    
    Key insight: l = a + c is liquid resources (what's left for consumption after 
    any pension deposit). This is different from total resources m = a + c + d.
    
    The grid par.grid_m represents liquid resources l in this context.
    
    CRITICAL: Marginal values v_l and v_b must be computed AFTER regridding
    using envelope theorem: v_l = w_a(b, a*) and v_b = w_b(b, a*) where a* = l - c*(b,l)
    """
    
    w = sol.w[t]
    wa = sol.wa[t]
    wb = sol.wb[t]

    # unpack outputs - these are on (b, l) grid where l = "liquid resources"
    inv_v = sol.inv_v_pure_c[t]
    c = sol.c_pure_c[t]
    v_l = sol.v_pure_c_m[t]  # Marginal value w.r.t. liquid resources
    v_b = sol.v_pure_c_b[t]  # Marginal value w.r.t. pension wealth

    for i_b in range(par.Nb_pd):
        b_val = par.grid_b_pd[i_b]
        
        # Build endogenous grid
        temp_c = np.zeros(par.Na_pd)
        temp_l = np.zeros(par.Na_pd)  # LIQUID resources, not total
        temp_w = np.zeros(par.Na_pd)
            
        # Invert Euler equation
        for i_a in range(par.Na_pd):
            temp_c[i_a] = utility.inv_marg_func(wa[i_b, i_a], par)
            temp_l[i_a] = par.grid_a_pd[i_a] + temp_c[i_a]  # l = a + c (liquid resources)
            temp_w[i_a] = w[i_b, i_a]
    
        # Standard upper envelope (no marginal tracking needed!)
        # Regrid from (a, l_endo) to (l_target) for this fixed b
        temp_v = np.zeros(par.Nm)
        
        segm_upperenvelope(
            temp_l,  # endogenous grid
            par.grid_a_pd,  # common grid (exogenous a)
            temp_c,  # policy on endogenous grid
            temp_w,  # post-decision value
            par.grid_m,  # target grid (liquid resources l)
            c[i_b, :],  # policy output
            temp_v,  # value output
            par
        )

        # Store value on (b, l) grid
        for i_l in range(par.Nm):
            inv_v[i_b, i_l] = -1.0 / temp_v[i_l]
            
            # ENVELOPE THEOREM: Compute marginal values from optimal choices
            # a*(b,l) = l - c*(b,l), then v_l = w_a(b, a*), v_b = w_b(b, a*)
            c_opt = c[i_b, i_l]
            l_val = par.grid_m[i_l]
            a_opt = l_val - c_opt
            
            # Safety: ensure valid consumption and assets
            if np.isnan(c_opt) or np.isnan(a_opt) or c_opt < 0 or a_opt < 0:
                v_l[i_b, i_l] = 1e-6
                v_b[i_b, i_l] = 1e-6
                continue
            
            # Interpolate w_a and w_b at (b, a_opt) using linear_interp
            # More robust than manual interpolation
            a_grid_1d = par.grid_a_pd
            wa_row = wa[i_b, :]
            wb_row = wb[i_b, :]
            
            # Clamp a_opt to grid bounds
            a_opt_clamped = max(a_grid_1d[0], min(a_grid_1d[-1], a_opt))
            
            # Use linear_interp from consav (returns the interpolated value)
            v_l_val = linear_interp.interp_1d(a_grid_1d, wa_row, a_opt_clamped)
            v_b_val = linear_interp.interp_1d(a_grid_1d, wb_row, a_opt_clamped)
            
            v_l[i_b, i_l] = v_l_val if v_l_val > 1e-10 else 1e-6
            v_b[i_b, i_l] = v_b_val if v_b_val > 1e-10 else 1e-6

@njit
def upperenvelope_1d_value(grid_endo, policy_vals, value_vals, grid_target, policy_out, value_out):
    """Simple 1D upper envelope for when values are already computed
    
    Args:
        grid_endo: Endogenous grid (can be non-monotonic)
        policy_vals: Policy function values on endogenous grid (1D or 2D array)
        value_vals: Value function on endogenous grid
        grid_target: Target exogenous grid
        policy_out: Output policy on target grid (1D or 2D array)
        value_out: Output value on target grid
    """
    N_endo = len(grid_endo)
    N_target = len(grid_target)
    is_2d = policy_vals.ndim == 2
    
    # Initialize
    value_out[:] = -np.inf
    if is_2d:
        policy_out[:, :] = 0.0
    else:
        policy_out[:] = 0.0
    
    # For each target point, find the best segment
    for i_tgt in range(N_target):
        target_val = grid_target[i_tgt]
        
        # Check all consecutive segments
        for i_endo in range(N_endo - 1):
            grid_low = grid_endo[i_endo]
            grid_high = grid_endo[i_endo + 1]
            
            # Skip if segment is degenerate
            if abs(grid_high - grid_low) < 1e-12:
                continue
            
            # Check if target is within this segment
            if (target_val >= min(grid_low, grid_high) and 
                target_val <= max(grid_low, grid_high)):
                
                # Linear interpolation
                weight = (target_val - grid_low) / (grid_high - grid_low)
                
                # Interpolate value
                v_interp = value_vals[i_endo] + weight * (value_vals[i_endo + 1] - value_vals[i_endo])
                
                # Update if this gives higher value
                if v_interp > value_out[i_tgt]:
                    value_out[i_tgt] = v_interp
                    
                    # Interpolate policy
                    if is_2d:
                        for j in range(policy_vals.shape[1]):
                            policy_out[i_tgt, j] = (policy_vals[i_endo, j] + 
                                weight * (policy_vals[i_endo + 1, j] - policy_vals[i_endo, j]))
                    else:
                        policy_out[i_tgt] = (policy_vals[i_endo] + 
                            weight * (policy_vals[i_endo + 1] - policy_vals[i_endo]))

@njit
def upperenvelope_1d_simple(grid_endo, c_vals, d_vals, v_vals, grid_target, c_out, d_out, v_out):
    """Simple 1D upper envelope - finds max value by checking all endogenous points
    
    For each target grid point, checks all segments of the endogenous grid
    and selects the one giving maximum value through linear interpolation.
    
    Args:
        grid_endo: endogenous grid (can be non-monotonic)
        c_vals, d_vals: policy functions on endogenous grid
        v_vals: value function on endogenous grid
        grid_target: target exogenous grid (monotonic)
        c_out, d_out, v_out: output arrays on target grid
    """
    n_endo = len(grid_endo)
    n_target = len(grid_target)
    
    # Initialize outputs
    for i in range(n_target):
        v_out[i] = -1e10
        c_out[i] = 0.0
        d_out[i] = 0.0
    
    # For each target point
    for i_tgt in range(n_target):
        target = grid_target[i_tgt]
        
        # Check all possible interpolations/extrapolations
        for i_endo in range(n_endo):
            # Try point value (nearest neighbor)
            if abs(grid_endo[i_endo] - target) < 1e-8:
                if v_vals[i_endo] > v_out[i_tgt]:
                    v_out[i_tgt] = v_vals[i_endo]
                    c_out[i_tgt] = c_vals[i_endo]
                    d_out[i_tgt] = d_vals[i_endo]
            
            # Try linear interpolation between consecutive points
            if i_endo < n_endo - 1:
                x0 = grid_endo[i_endo]
                x1 = grid_endo[i_endo + 1]
                
                # Check if target is in this segment
                if (x0 <= target <= x1) or (x1 <= target <= x0):
                    if abs(x1 - x0) > 1e-12:
                        weight = (target - x0) / (x1 - x0)
                        weight = max(0.0, min(1.0, weight))  # Clamp to [0,1]
                        
                        v_interp = (1.0 - weight) * v_vals[i_endo] + weight * v_vals[i_endo + 1]
                        
                        if v_interp > v_out[i_tgt]:
                            v_out[i_tgt] = v_interp
                            c_out[i_tgt] = (1.0 - weight) * c_vals[i_endo] + weight * c_vals[i_endo + 1]
                            d_out[i_tgt] = (1.0 - weight) * d_vals[i_endo] + weight * d_vals[i_endo + 1]
        
        # If no good value found, use nearest neighbor
        if v_out[i_tgt] < -1e9:
            i_nearest = 0
            dist_min = 1e10
            for i_endo in range(n_endo):
                dist = abs(grid_endo[i_endo] - target)
                if dist < dist_min:
                    dist_min = dist
                    i_nearest = i_endo
            
            v_out[i_tgt] = v_vals[i_nearest]
            c_out[i_tgt] = c_vals[i_nearest]
            d_out[i_tgt] = d_vals[i_nearest]

@njit
def inv_mn_and_v(c, d, a, b, w, par):
    """Compute endogenous (m,n) and value from choices and post-decision states"""
    v = utility.func(c, par) + w
    m = a + c + d
    n = b - d - pens.func(d, par)
    return m, n, v

@njit
def solve_pension_egm(t, sol, par):
    """Sequential EGM with correct liquid resources formulation
    
    Subproblem 1 (Inner): Pure consumption EGM
      - Exo: (b×, a×) → Endo: l → Regrid to → Exo: (b×, l×)
      - Output: c(b×, l×), v(b×, l×), v_l(b×, l×), v_b(b×, l×)
      where l is LIQUID resources (l = a + c)
    
    Subproblem 2 (Outer): Pure pension EGM  
      - Exo: (b×, l×) → Invert FOC for d → Endo: (n, m)
      - Where n = b - d - ψ(d) and m = l + d
      - Both n and m are endogenous!
      - Output: d(n×, m×), c(n×, m×), V(n×, m×)
    
    Key insight: (b, l) represents "post-choice" state where l is liquid 
    resources available for consumption. We invert to find what d led to this.
    """
    
    # SUBPROBLEM 1: Pure consumption EGM on (b×, a×) → (b×, l×)
    solve_pure_c(t, sol, par)
    # Now sol.c_pure_c[t], sol.v_pure_c_m[t], sol.v_pure_c_b[t] are defined on (b×, l×) grid
    # where l = liquid resources
    
    # SUBPROBLEM 2: Pure pension EGM on (b×, l×) → (n×, m×)
    # Key: We now work on (b, l) grid and compute BOTH n and m as endogenous
    
    c_pure_c = sol.c_pure_c[t]  # On (b, l) grid
    v_l = sol.v_pure_c_m[t]     # Marginal value w.r.t. liquid resources
    v_b = sol.v_pure_c_b[t]     # Marginal value w.r.t. pension wealth
    
    # Output arrays on final (n×, m×) grid
    inv_v = sol.inv_v[t]
    inv_vm = sol.inv_vm[t]
    c = sol.c[t]
    d = sol.d[t]
    
    # Build endogenous grid from (b, l) → d → (n, m)
    # We need to flatten the 2D (b, l) grid since both n and m will be endogenous
    N_bl = par.Nb_pd * par.Nm
    
    m_endo_flat = np.zeros(N_bl)
    n_endo_flat = np.zeros(N_bl)
    c_endo_flat = np.zeros(N_bl)
    d_endo_flat = np.zeros(N_bl)
    v_endo_flat = np.zeros(N_bl)
    
    idx = 0
    for i_b in range(par.Nb_pd):
        b_val = par.grid_b_pd[i_b]
        for i_l in range(par.Nm):
            l_val = par.grid_m[i_l]  # l is on grid_m
            
            # Get values from Subproblem 1
            c_val = c_pure_c[i_b, i_l]
            v_l_val = v_l[i_b, i_l]
            v_b_val = v_b[i_b, i_l]
            inv_v_val = sol.inv_v_pure_c[t][i_b, i_l]
            if abs(inv_v_val) < 1e-10:
                v_val = -1e10
            else:
                v_val = -1.0 / inv_v_val
            
            # Invert pension FOC at this (b, l) point
            # FOC: ψ'(d) = v_l / v_b - 1
            # For ψ(d) = χ log(1 + d), we have ψ'(d) = χ/(1 + d)
            # So: d = χ * v_b / (v_l - v_b) - 1
            
            denom = v_l_val - v_b_val
            
            # Default to zero contribution
            d_val = 0.0
            
            # Only compute d if denominator is safely positive
            if v_b_val > 1e-10 and denom > 1e-10:
                # Invert FOC: d = χ * v_b / (v_l - v_b) - 1
                # This formula comes from: ψ'(d) = v_l/v_b - 1
                # where ψ'(d) = χ/(1+d) for ψ(d) = χ log(1+d)
                
                ratio = (par.chi * v_b_val) / denom
                d_val = ratio - 1.0
                
                # CRITICAL: Ensure d > -1 BEFORE any use in pens.func
                # log(1 + d) requires d > -1
                if d_val < -0.99:
                    d_val = 0.0
                elif d_val < 0.0:
                    d_val = 0.0  # Clamp negative values to zero (no withdrawal)
                else:
                    # Cap at reasonable maximum to avoid numerical issues
                    d_val = min(d_val, min(b_val, l_val, 100.0))
            
            # Compute BOTH endogenous states
            # At this point d_val >= 0, so log(1 + d_val) is safe
            psi_d = pens.func(d_val, par)  # This is safe now
            n_val = b_val - d_val - psi_d   # Endogenous n
            m_val = l_val + d_val            # Endogenous m (total resources)
            
            # Store in flattened arrays
            m_endo_flat[idx] = m_val
            n_endo_flat[idx] = n_val
            c_endo_flat[idx] = c_val
            d_endo_flat[idx] = d_val
            v_endo_flat[idx] = v_val
            
            idx += 1
    
    # Reshape for 2D upper envelope (G2EGM's upper envelope)
    # Manual reshape since Numba reshape can be problematic
    m_endo_2d = np.zeros((par.Nb_pd, par.Nm))
    n_endo_2d = np.zeros((par.Nb_pd, par.Nm))
    c_endo_2d = np.zeros((par.Nb_pd, par.Nm))
    d_endo_2d = np.zeros((par.Nb_pd, par.Nm))
    v_endo_2d = np.zeros((par.Nb_pd, par.Nm))
    
    idx = 0
    for i_b in range(par.Nb_pd):
        for i_l in range(par.Nm):
            # Validate and sanitize values
            m_val = m_endo_flat[idx]
            n_val = n_endo_flat[idx]
            c_val = c_endo_flat[idx]
            d_val = d_endo_flat[idx]
            v_val = v_endo_flat[idx]
            
            # Check for NaN or inf
            if np.isnan(m_val) or np.isinf(m_val):
                m_val = 0.0
            if np.isnan(n_val) or np.isinf(n_val):
                n_val = 0.0
            if np.isnan(c_val) or np.isinf(c_val):
                c_val = 0.01
            if np.isnan(d_val) or np.isinf(d_val):
                d_val = 0.0
            if np.isnan(v_val) or np.isinf(v_val) or v_val > 0:
                v_val = -1e10
            
            m_endo_2d[i_b, i_l] = m_val
            n_endo_2d[i_b, i_l] = n_val
            c_endo_2d[i_b, i_l] = c_val
            d_endo_2d[i_b, i_l] = d_val
            v_endo_2d[i_b, i_l] = v_val
            idx += 1
    
    # Use G2EGM's 2D upper envelope
    v_out = np.zeros((par.Nn, par.Nm))
    w = sol.w[t]  # Post-decision value function
    
    ue_local.compute(
        c, d, v_out,
        m_endo_2d, n_endo_2d,
        c_endo_2d, d_endo_2d, v_endo_2d,
        1, w, par
    )
    
    # Convert to inverse value and compute marginal value
    for i_n in range(par.Nn):
        for i_m in range(par.Nm):
            if v_out[i_n, i_m] > 1e-10:
                inv_v[i_n, i_m] = -1.0 / v_out[i_n, i_m]
            else:
                inv_v[i_n, i_m] = -1e10
            
            vm_val = utility.marg_func(c[i_n, i_m], par)
            if vm_val > 1e-10:
                inv_vm[i_n, i_m] = 1.0 / vm_val
            else:
                inv_vm[i_n, i_m] = 1e10
