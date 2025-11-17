"""SEGM - Sequential Endogenous Grid Method with numba acceleration

SEGM applies EGM twice sequentially:
1. Pure Consumption EGM: (b, a) → (b, l) where l = liquid resources (reuse NEGM)
2. Pure Pension EGM: (b, l) → (n, m) by inverting pension FOC

This replaces NEGM's VFI outer loop with a second EGM step.
"""

import numpy as np
from numba import njit

# consav
from consav import linear_interp

# local modules
import utility
import pens
import NEGM  # Reuse NEGM's pure consumption solver

@njit
def upperenvelope_2d_segm(n_endo_2d, m_endo_2d, c_endo_2d, d_endo_2d, v_endo_2d,
                          grid_n, grid_m, c_out, d_out, v_out, w, par):
    """
    Robust 2D upper envelope for SEGM (matches quality of upperenvelope.py)
    
    Given endogenous grids from pension EGM, regrid to common (n, m) grid.
    Key difference from G2EGM: c is predetermined, only choose best d.
    
    Args:
        n_endo_2d, m_endo_2d: endogenous states [Nb, Nl]
        c_endo_2d, d_endo_2d, v_endo_2d: policies and values [Nb, Nl]
        grid_n, grid_m: target exogenous grids
        c_out, d_out, v_out: output arrays [Nn, Nm]
        par: parameters (for extrapolation control)
    """
    Nb, Nl = n_endo_2d.shape
    Nn = len(grid_n)
    Nm = len(grid_m)
    
    # Initialize outputs (use safe defaults for unfilled regions)
    for i_n in range(Nn):
        for i_m in range(Nm):
            c_out[i_n, i_m] = 0.0
            d_out[i_n, i_m] = 0.0
            v_out[i_n, i_m] = -np.inf
    
    # Rigorous validation (matches upperenvelope.py)
    valid = np.ones((Nb, Nl), dtype=np.bool_)
    for i_b in range(Nb):
        for i_l in range(Nl):
            valid[i_b, i_l] &= np.imag(c_endo_2d[i_b, i_l]) == 0
            valid[i_b, i_l] &= np.imag(d_endo_2d[i_b, i_l]) == 0
            valid[i_b, i_l] &= ~np.isnan(v_endo_2d[i_b, i_l])
            valid[i_b, i_l] &= c_endo_2d[i_b, i_l] >= -0.5
            valid[i_b, i_l] &= d_endo_2d[i_b, i_l] >= -0.5
            valid[i_b, i_l] &= m_endo_2d[i_b, i_l] > -0.1
            valid[i_b, i_l] &= n_endo_2d[i_b, i_l] > -0.2  # Allow slightly negative n (will be clamped)
            valid[i_b, i_l] &= m_endo_2d[i_b, i_l] < par.m_max + 3  # Allow more headroom for l+d
            valid[i_b, i_l] &= n_endo_2d[i_b, i_l] < par.n_max + 1
    
    valid_count = 0
    for i_b in range(Nb):
        for i_l in range(Nl):
            if valid[i_b, i_l]:
                valid_count += 1
    
    if valid_count < 100:
        return
    
    # Track holes for filling (like upperenvelope.py)
    holes = np.ones((Nn, Nm))
    
    # Form triangles and interpolate
    for i_b in range(Nb):
        for i_l in range(Nl):
            # Two triangles per grid cell
            for tri in range(2):
                _process_triangle(i_l, i_b, tri, 
                                  m_endo_2d, n_endo_2d, c_endo_2d, d_endo_2d, v_endo_2d,
                                  Nl, Nb, valid, grid_n, grid_m,
                                  c_out, d_out, v_out, holes, w, par)
    
    # Fill holes (grid points not covered by any triangle)
    _fill_holes(c_out, d_out, v_out, holes, grid_n, grid_m)

@njit
def _process_triangle(i_l, i_b, tri, m, n, c, d, v, Nl, Nb, valid,
                      grid_n, grid_m, c_out, d_out, v_out, holes, w, par):
    """Process one triangle (matches upperenvelope.py logic)"""
    
    # a. Define simplex in (b, l) space
    i_b_1, i_l_1 = i_b, i_l
    
    if i_b == Nb - 1:
        return
    i_b_2, i_l_2 = i_b + 1, i_l
    
    i_b_3, i_l_3 = -1, -1
    
    if tri == 0:
        if i_l == 0 or i_b == Nb - 1:
            return
        i_b_3, i_l_3 = i_b + 1, i_l - 1
    else:
        if i_l == Nl - 1:
            return
        i_b_3, i_l_3 = i_b, i_l + 1
    
    if not (valid[i_b_1, i_l_1] and valid[i_b_2, i_l_2] and valid[i_b_3, i_l_3]):
        return
    
    # b. Simplex in (m, n) space
    m1 = m[i_b_1, i_l_1]
    m2 = m[i_b_2, i_l_2]
    m3 = m[i_b_3, i_l_3]
    
    n1 = n[i_b_1, i_l_1]
    n2 = n[i_b_2, i_l_2]
    n3 = n[i_b_3, i_l_3]
    
    # c. Bounding box with binary search (like upperenvelope.py)
    m_max = np.fmax(m1, np.fmax(m2, m3))
    m_min = np.fmin(m1, np.fmin(m2, m3))
    n_max = np.fmax(n1, np.fmax(n2, n3))
    n_min = np.fmin(n1, np.fmin(n2, n3))
    
    im_low = 0
    if m_min >= 0:
        im_low = linear_interp.binary_search(0, len(grid_m), grid_m, m_min)
    im_high = linear_interp.binary_search(0, len(grid_m), grid_m, m_max) + 1
    
    in_low = 0
    if n_min >= 0:
        in_low = linear_interp.binary_search(0, len(grid_n), grid_n, n_min)
    in_high = linear_interp.binary_search(0, len(grid_n), grid_n, n_max) + 1
    
    # Extrapolation control (like upperenvelope.py)
    im_low = max(im_low - par.egm_extrap_add, 0)
    im_high = min(im_high + par.egm_extrap_add, len(grid_m))
    in_low = max(in_low - par.egm_extrap_add, 0)
    in_high = min(in_high + par.egm_extrap_add, len(grid_n))
    
    # d. Barycentric interpolation
    denom = (n2 - n3) * (m1 - m3) + (m3 - m2) * (n1 - n3)
    if abs(denom) < 1e-10:
        return
    
    # e. Loop through target grid
    for i_n in range(in_low, in_high):
        for i_m in range(im_low, im_high):
            
            m_now = grid_m[i_m]
            n_now = grid_n[i_n]
            
            # Barycentric coordinates
            w1 = ((n2 - n3) * (m_now - m3) + (m3 - m2) * (n_now - n3)) / denom
            w2 = ((n3 - n1) * (m_now - m3) + (m1 - m3) * (n_now - n3)) / denom
            w3 = 1.0 - w1 - w2
            
            # Extrapolation control (like upperenvelope.py)
            if w1 < par.egm_extrap_w or w2 < par.egm_extrap_w or w3 < par.egm_extrap_w:
                continue
            
            # Interpolate choices - SEGM: c predetermined, d varies
            c_interp = w1 * c[i_b_1, i_l_1] + w2 * c[i_b_2, i_l_2] + w3 * c[i_b_3, i_l_3]
            d_interp = w1 * d[i_b_1, i_l_1] + w2 * d[i_b_2, i_l_2] + w3 * d[i_b_3, i_l_3]
            
            # Compute implied post-decision states
            a_interp = m_now - c_interp - d_interp
            b_interp = n_now + d_interp + pens.func(d_interp, par)
            
            # Feasibility checks
            if c_interp <= 0.0 or d_interp < 0.0 or a_interp < 0.0 or b_interp < 0.0:
                continue
            
            # Recompute value (like G2EGM does) for consistency
            w_interp = linear_interp.interp_2d(par.grid_b_pd, par.grid_a_pd, w, b_interp, a_interp)
            v_interp = utility.func(c_interp, par) + pens.func(d_interp, par) + w_interp
            
            # Upper envelope: update if better
            if v_interp > v_out[i_n, i_m]:
                v_out[i_n, i_m] = v_interp
                c_out[i_n, i_m] = c_interp
                d_out[i_n, i_m] = d_interp
                holes[i_n, i_m] = 0

@njit
def _fill_holes(c_out, d_out, v_out, holes, grid_n, grid_m):
    """Fill holes using nearby valid points (matches upperenvelope.py logic)"""
    
    Nn = len(grid_n)
    Nm = len(grid_m)
    
    # a. Locate global bounding box with content
    i_n_min = 0
    i_n_max = Nn - 1
    min_n = np.inf
    max_n = -np.inf
    
    i_m_min = 0
    i_m_max = Nm - 1
    min_m = np.inf
    max_m = -np.inf
    
    for i_n in range(Nn):
        for i_m in range(Nm):
            
            m_now = grid_m[i_m]
            n_now = grid_n[i_n]
            
            if holes[i_n, i_m] == 1:
                continue
            
            if m_now < min_m:
                min_m = m_now
                i_m_min = i_m
            
            if m_now > max_m:
                max_m = m_now
                i_m_max = i_m
            
            if n_now < min_n:
                min_n = n_now
                i_n_min = i_n
            
            if n_now > max_n:
                max_n = n_now
                i_n_max = i_n
    
    # b. Fill holes within bounding box
    i_n_max = min(i_n_max + 1, Nn)
    i_m_max = min(i_m_max + 1, Nm)
    
    for i_n in range(i_n_min, i_n_max):
        for i_m in range(i_m_min, i_m_max):
            
            if holes[i_n, i_m] == 0:  # Not a hole
                continue
            
            # Search window
            m_add = 2
            n_add = 2
            
            i_n_close_min = max(0, i_n - n_add)
            i_n_close_max = min(i_n + n_add + 1, Nn)
            i_m_close_min = max(0, i_m - m_add)
            i_m_close_max = min(i_m + m_add + 1, Nm)
            
            # Find best nearby point
            for i_n_close in range(i_n_close_min, i_n_close_max):
                for i_m_close in range(i_m_close_min, i_m_close_max):
                    
                    if holes[i_n_close, i_m_close] == 1:  # Also a hole
                        continue
                    
                    # Copy policy from nearby valid point
                    c_interp = c_out[i_n_close, i_m_close]
                    d_interp = d_out[i_n_close, i_m_close]
                    v_interp = v_out[i_n_close, i_m_close]
                    
                    # Update if better
                    if v_interp > v_out[i_n, i_m]:
                        v_out[i_n, i_m] = v_interp
                        c_out[i_n, i_m] = c_interp
                        d_out[i_n, i_m] = d_interp

@njit
def solve(t, sol, par):
    """
    SEGM solver for one period
    
    Two-stage Sequential EGM (reuses NEGM's pure_c, adds pension EGM):
    1. Pure Consumption EGM: (b, a) → (b, l) where l = liquid resources [REUSE NEGM]
    2. Pure Pension EGM: (b, l) → (n, m) by inverting pension FOC [NEW]
    """
    
    # Unpack post-decision value function and marginals
    w = sol.w[t]
    wa = sol.wa[t]
    wb = sol.wb[t]
    
    # Intermediate arrays for subproblem 1 output
    c_pure_c = sol.c_pure_c[t]
    v_l = sol.v_pure_c_l[t]  # Marginal value w.r.t. liquid resources l
    v_b = sol.v_pure_c_b[t]  # Marginal value w.r.t. pension wealth b
    
    # =========================================================================
    # SUBPROBLEM 1: Pure Consumption (b, a) → (b, l) [REUSE NEGM]
    # =========================================================================
    
    # Call NEGM's pure consumption solver - it already does everything we need!
    # It computes c(b, l) where l = liquid resources = a + c
    NEGM.solve_pure_c(t, sol, par)
    
    # Envelope theorem: SEGM uses same FOCs as G2EGM, just sequentially!
    # After solving consumption, we have v_l = wa and v_b = wb at optimal a* = l - c*
    # The pension FOC is: ψ'(d) = v_l/v_b - 1 = wa/wb - 1 (same as G2EGM!)
    for i_b in range(par.Nb_pd):
        b_val = par.grid_b_pd[i_b]
        for i_l in range(par.Nm):
            c_opt = c_pure_c[i_b, i_l]
            l_val = par.grid_m[i_l]  # NEGM outputs on grid_l which equals grid_m
            a_opt = l_val - c_opt
            
            # Safety checks
            if c_opt < 1e-8 or a_opt < 0:
                v_l[i_b, i_l] = 1e10
                v_b[i_b, i_l] = 1e-6
                continue
            
            # By envelope: v_l = wa(b, a*) and v_b = wb(b, a*) at optimal a*
            a_clamped = max(par.grid_a_pd[0], min(par.grid_a_pd[-1], a_opt))
            v_l[i_b, i_l] = linear_interp.interp_2d(par.grid_b_pd, par.grid_a_pd, wa, b_val, a_clamped)
            v_b[i_b, i_l] = linear_interp.interp_2d(par.grid_b_pd, par.grid_a_pd, wb, b_val, a_clamped)
            
            # Ensure positive  
            if v_l[i_b, i_l] < 1e-10:
                v_l[i_b, i_l] = 1e10
            if v_b[i_b, i_l] < 1e-10:
                v_b[i_b, i_l] = 1e-6
    
    # =========================================================================
    # SUBPROBLEM 2: Pure Pension EGM (b, l) → (n, m)
    # =========================================================================
    
    # Build endogenous grid in 2D structure
    Nb = par.Nb_pd
    Nl = par.Nm
    
    d_endo_2d = np.zeros((Nb, Nl))
    c_endo_2d = np.zeros((Nb, Nl))
    n_endo_2d = np.zeros((Nb, Nl))
    m_endo_2d = np.zeros((Nb, Nl))
    v_endo_2d = np.zeros((Nb, Nl))
    
    # Initialize with invalid values
    for i_b in range(Nb):
        for i_l in range(Nl):
            d_endo_2d[i_b, i_l] = np.nan
            c_endo_2d[i_b, i_l] = np.nan
            n_endo_2d[i_b, i_l] = np.nan
            m_endo_2d[i_b, i_l] = np.nan
            v_endo_2d[i_b, i_l] = -np.inf
    
    for i_b in range(Nb):
        b_val = par.grid_b_pd[i_b]
        
        for i_l in range(Nl):
            l_val = par.grid_m[i_l]
            
            c_val = c_pure_c[i_b, i_l]
            v_l_val = v_l[i_b, i_l]
            v_b_val = v_b[i_b, i_l]
            
            # Skip if c is invalid
            if c_val <= 1e-8:
                continue
            
            # Invert pension FOC: ψ'(d) = (v_l - v_b)/v_b
            # For ψ(d) = χ log(1 + d): χ/(1+d) = (v_l - v_b)/v_b
            # Rearranging: χ v_b/(v_l - v_b) = 1 + d
            # Therefore: d = (χ v_b)/(v_l - v_b) - 1
            
            denom = v_l_val - v_b_val
            
            # Try unconstrained solution
            d_best = 0.0
            v_best = -np.inf
            
            if denom > 1e-6 and v_b_val > 1e-10:
                d_test = (par.chi * v_b_val) / denom - 1.0
                
                if d_test > 0:
                    d_test = min(d_test, b_val, l_val)
                    
                    # Compute endogenous (n, m)
                    psi_val = pens.func(d_test, par)
                    n_test = b_val - d_test - psi_val
                    m_test = l_val + d_test
                    a_test = m_test - c_val
                    
                    # Check validity (relaxed constraints for boundary extrapolation)
                    n_test_clamped = max(n_test, 0.0)  # Allow slight negative n, clamp to 0
                    if (n_test_clamped >= par.grid_n[0] and m_test >= par.grid_m[0] and
                        a_test >= par.grid_a_pd[0] - 1.0 and a_test <= par.grid_a_pd[-1] + 1.0):
                        # Clamp a for interpolation (allow extrapolation by 1 unit)
                        a_interp = max(par.grid_a_pd[0], min(par.grid_a_pd[-1], a_test))
                        
                        # Interpolate w(n_test, a_test)
                        w_val = linear_interp.interp_2d(
                            par.grid_b_pd, par.grid_a_pd, w, n_test_clamped, a_interp
                        )
                        v_test = utility.func(c_val, par) + pens.func(d_test, par) + w_val
                        
                        if v_test > v_best:
                            d_best = d_test
                            v_best = v_test
            
            # Always try d=0 (constraint-binding)
            d_test = 0.0
            psi_val = pens.func(d_test, par)
            n_test = b_val - d_test - psi_val
            m_test = l_val + d_test
            a_test = m_test - c_val
            
            # Check validity (relaxed constraints)
            n_test_clamped = max(n_test, 0.0)
            if (n_test_clamped >= par.grid_n[0] and m_test >= par.grid_m[0] and
                a_test >= par.grid_a_pd[0] - 1.0 and a_test <= par.grid_a_pd[-1] + 1.0):
                # Clamp a for interpolation
                a_interp = max(par.grid_a_pd[0], min(par.grid_a_pd[-1], a_test))
                
                w_val = linear_interp.interp_2d(
                    par.grid_b_pd, par.grid_a_pd, w, n_test_clamped, a_interp
                )
                v_test = utility.func(c_val, par) + pens.func(d_test, par) + w_val
                
                if v_test > v_best:
                    d_best = 0.0
                    v_best = v_test
            
            # Store the best solution
            if v_best > -np.inf:
                psi_val = pens.func(d_best, par)
                d_endo_2d[i_b, i_l] = d_best
                n_endo_2d[i_b, i_l] = b_val - d_best - psi_val
                m_endo_2d[i_b, i_l] = l_val + d_best
                c_endo_2d[i_b, i_l] = c_val
                v_endo_2d[i_b, i_l] = v_best
    
    # =========================================================================
    # Regrid using SEGM's custom 2D upper envelope
    # =========================================================================
    
    # Diagnostic: count valid endogenous points
    valid_count = 0
    d_positive_count = 0
    for i_b in range(Nb):
        for i_l in range(Nl):
            if not np.isnan(d_endo_2d[i_b, i_l]) and not np.isinf(v_endo_2d[i_b, i_l]):
                valid_count += 1
                if d_endo_2d[i_b, i_l] > 0.01:
                    d_positive_count += 1
    
    # If too few valid points, skip regridding (will use defaults)
    if valid_count < 100:
        return
    
    c_out = sol.c[t]
    d_out = sol.d[t]
    v_out = np.zeros((par.Nn, par.Nm))
    
    upperenvelope_2d_segm(
        n_endo_2d, m_endo_2d, c_endo_2d, d_endo_2d, v_endo_2d,
        par.grid_n, par.grid_m, c_out, d_out, v_out, w, par
    )
    
    # Convert to inverse value and compute marginals
    inv_v = sol.inv_v[t]
    inv_vm = sol.inv_vm[t]
    inv_vn = sol.inv_vn[t]
    
    for i_n in range(par.Nn):
        for i_m in range(par.Nm):
            # Value functions should be negative
            if v_out[i_n, i_m] < -1e-10 and not np.isinf(v_out[i_n, i_m]):
                inv_v[i_n, i_m] = -1.0 / v_out[i_n, i_m]
            else:
                inv_v[i_n, i_m] = -1e10
            
            # Marginal value w.r.t. m
            vm_val = utility.marg_func(c_out[i_n, i_m], par)
            if vm_val > 1e-10:
                inv_vm[i_n, i_m] = 1.0 / vm_val
            else:
                inv_vm[i_n, i_m] = 1e10
    
    # Compute marginal value w.r.t. n using envelope theorem (like G2EGM)
    # vn = wb(b*, a*) where a* = m - c - d and b* = n + d + ψ(d)
    for i_n in range(par.Nn):
        n_val = par.grid_n[i_n]
        for i_m in range(par.Nm):
            m_val = par.grid_m[i_m]
            c_val = c_out[i_n, i_m]
            d_val = d_out[i_n, i_m]
            
            # Compute optimal post-decision states
            a_val = m_val - c_val - d_val
            b_val = n_val + d_val + pens.func(d_val, par)
            
            # By envelope theorem: vn = wb(b*, a*)
            if (a_val >= par.grid_a_pd[0] and a_val <= par.grid_a_pd[-1] and
                b_val >= par.grid_b_pd[0] and b_val <= par.grid_b_pd[-1]):
                vn_val = linear_interp.interp_2d(par.grid_b_pd, par.grid_a_pd, wb, b_val, a_val)
                if abs(vn_val) > 1e-10:
                    inv_vn[i_n, i_m] = 1.0 / vn_val
                else:
                    inv_vn[i_n, i_m] = 1e10 if vn_val >= 0 else -1e10
            else:
                # Out of bounds - use safe default
                inv_vn[i_n, i_m] = 1e10
