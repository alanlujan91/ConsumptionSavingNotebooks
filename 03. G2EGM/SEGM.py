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
                          grid_n, grid_m, c_out, d_out, v_out):
    """
    2D upper envelope for SEGM - triangulation with barycentric interpolation
    
    Args:
        n_endo_2d, m_endo_2d: endogenous states [Nb, Nl]
        c_endo_2d, d_endo_2d, v_endo_2d: policies and values [Nb, Nl]
        grid_n, grid_m: target exogenous grids
        c_out, d_out, v_out: output arrays [Nn, Nm]
    """
    Nb, Nl = n_endo_2d.shape
    Nn = len(grid_n)
    Nm = len(grid_m)
    
    # Initialize outputs
    for i_n in range(Nn):
        for i_m in range(Nm):
            c_out[i_n, i_m] = 0.0
            d_out[i_n, i_m] = 0.0
            v_out[i_n, i_m] = -np.inf
    
    # Mark valid points
    valid = np.ones((Nb, Nl), dtype=np.bool_)
    for i_b in range(Nb):
        for i_l in range(Nl):
            valid[i_b, i_l] = (not np.isnan(v_endo_2d[i_b, i_l]) and
                              not np.isinf(v_endo_2d[i_b, i_l]) and
                              c_endo_2d[i_b, i_l] > 1e-8 and
                              m_endo_2d[i_b, i_l] > -0.1 and
                              n_endo_2d[i_b, i_l] > -0.1)
    
    valid_count = 0
    for i_b in range(Nb):
        for i_l in range(Nl):
            if valid[i_b, i_l]:
                valid_count += 1
    
    if valid_count < 100:
        return
    
    # Form triangles and interpolate
    for i_b in range(Nb):
        for i_l in range(Nl):
            # Two triangles per grid cell
            for tri in range(2):
                _process_triangle(i_l, i_b, tri, 
                                 m_endo_2d, n_endo_2d, c_endo_2d, d_endo_2d, v_endo_2d,
                                 Nl, Nb, valid, grid_n, grid_m,
                                 c_out, d_out, v_out)
    
    # Fill holes (grid points not covered by any triangle)
    _fill_holes(c_out, d_out, v_out, grid_n, grid_m)

@njit
def _process_triangle(i_l, i_b, tri, m, n, c, d, v, Nl, Nb, valid,
                     grid_n, grid_m, c_out, d_out, v_out):
    """Process one triangle for upper envelope"""
    
    # Define triangle vertices
    i_b_1, i_l_1 = i_b, i_l
    
    if i_b == Nb-1:
        return
    i_b_2, i_l_2 = i_b+1, i_l
    
    if tri == 0:
        if i_l == 0 or i_b == Nb-1:
            return
        i_b_3, i_l_3 = i_b+1, i_l-1
    else:
        if i_l == Nl-1:
            return
        i_b_3, i_l_3 = i_b, i_l+1
    
    # Check validity
    if not (valid[i_b_1, i_l_1] and valid[i_b_2, i_l_2] and valid[i_b_3, i_l_3]):
        return
    
    # Get triangle vertices in (m, n) space
    m1, m2, m3 = m[i_b_1, i_l_1], m[i_b_2, i_l_2], m[i_b_3, i_l_3]
    n1, n2, n3 = n[i_b_1, i_l_1], n[i_b_2, i_l_2], n[i_b_3, i_l_3]
    
    # Bounding box
    m_min = min(m1, min(m2, m3))
    m_max = max(m1, max(m2, m3))
    n_min = min(n1, min(n2, n3))
    n_max = max(n1, max(n2, n3))
    
    # Find grid indices using manual search
    im_low = 0
    for i in range(len(grid_m)):
        if grid_m[i] >= m_min:
            im_low = i
            break
    
    im_high = len(grid_m)
    for i in range(len(grid_m)-1, -1, -1):
        if grid_m[i] <= m_max:
            im_high = i + 1
            break
    
    in_low = 0
    for i in range(len(grid_n)):
        if grid_n[i] >= n_min:
            in_low = i
            break
    
    in_high = len(grid_n)
    for i in range(len(grid_n)-1, -1, -1):
        if grid_n[i] <= n_max:
            in_high = i + 1
            break
    
    im_low = max(0, im_low)
    im_high = min(len(grid_m), im_high)
    in_low = max(0, in_low)
    in_high = min(len(grid_n), in_high)
    
    # Barycentric interpolation denominator
    denom = (n2-n3)*(m1-m3) + (m3-m2)*(n1-n3)
    if abs(denom) < 1e-10:
        return
    
    # Loop through grid points in bounding box
    for i_n in range(in_low, in_high):
        n_val = grid_n[i_n]
        for i_m in range(im_low, im_high):
            m_val = grid_m[i_m]
            
            # Barycentric coordinates
            w1 = ((n2-n3)*(m_val-m3) + (m3-m2)*(n_val-n3)) / denom
            w2 = ((n3-n1)*(m_val-m3) + (m1-m3)*(n_val-n3)) / denom
            w3 = 1.0 - w1 - w2
            
            # Check if point is inside triangle (with small tolerance)
            if w1 >= -1e-8 and w2 >= -1e-8 and w3 >= -1e-8:
                # Interpolate values
                c_val = w1*c[i_b_1, i_l_1] + w2*c[i_b_2, i_l_2] + w3*c[i_b_3, i_l_3]
                d_val = w1*d[i_b_1, i_l_1] + w2*d[i_b_2, i_l_2] + w3*d[i_b_3, i_l_3]
                v_val = w1*v[i_b_1, i_l_1] + w2*v[i_b_2, i_l_2] + w3*v[i_b_3, i_l_3]
                
                # Upper envelope: keep if higher value
                if v_val > v_out[i_n, i_m]:
                    c_out[i_n, i_m] = c_val
                    d_out[i_n, i_m] = d_val
                    v_out[i_n, i_m] = v_val

@njit
def _fill_holes(c_out, d_out, v_out, grid_n, grid_m):
    """Fill holes (grid points not covered by triangulation) using nearest neighbor"""
    Nn, Nm = c_out.shape
    
    for i_n in range(Nn):
        for i_m in range(Nm):
            if np.isnan(c_out[i_n, i_m]) or np.isinf(v_out[i_n, i_m]) or v_out[i_n, i_m] >= -1e-10:
                # Find nearest valid neighbor (expanded search radius)
                best_dist = np.inf
                best_c = grid_m[i_m]  # Default: consume all resources
                best_d = 0.0           # Default: no pension
                best_v = -1e10
                
                # Expand search radius
                for radius in range(1, min(20, max(Nn, Nm))):
                    i_n_min = max(0, i_n-radius)
                    i_n_max = min(Nn, i_n+radius+1)
                    i_m_min = max(0, i_m-radius)
                    i_m_max = min(Nm, i_m+radius+1)
                    
                    for j_n in range(i_n_min, i_n_max):
                        for j_m in range(i_m_min, i_m_max):
                            if (not np.isnan(c_out[j_n, j_m]) and 
                                not np.isinf(v_out[j_n, j_m]) and 
                                v_out[j_n, j_m] < -1e-10):
                                dist = (grid_n[i_n] - grid_n[j_n])**2 + (grid_m[i_m] - grid_m[j_m])**2
                                if dist < best_dist:
                                    best_dist = dist
                                    best_c = c_out[j_n, j_m]
                                    best_d = d_out[j_n, j_m]
                                    best_v = v_out[j_n, j_m]
                    
                    # If found a valid neighbor, stop searching
                    if best_dist < np.inf:
                        break
                
                c_out[i_n, i_m] = best_c
                d_out[i_n, i_m] = best_d
                v_out[i_n, i_m] = best_v

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
    
    # Now compute marginal values using envelope theorem
    # v_l(b, l) = u'(c*(b, l)) - marginal utility of consumption
    # v_b(b, l) = wb(b, a*) where a* = l - c*(b, l)
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
            
            # v_l = u'(c) - marginal utility w.r.t. liquid resources (same as u'(c))
            v_l[i_b, i_l] = utility.marg_func(c_opt, par)
            
            # v_b = wb(b, a*) - interpolate from post-decision marginal value
            a_clamped = max(par.grid_a_pd[0], min(par.grid_a_pd[-1], a_opt))
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
            
            # Invert pension FOC: ψ'(d) = v_l - v_b
            # For ψ(d) = χ log(1 + d): χ/(1+d) = v_l - v_b
            # Rearranging: d = χ/(v_l - v_b) - 1
            
            denom = v_l_val - v_b_val
            
            # Try unconstrained solution
            d_best = 0.0
            v_best = -np.inf
            
            if denom > 1e-6 and v_b_val > 1e-10:
                d_test = par.chi / denom - 1.0
                
                if d_test > 0:
                    d_test = min(d_test, b_val, l_val)
                    
                    # Compute endogenous (n, m)
                    psi_val = pens.func(d_test, par)
                    n_test = b_val - d_test - psi_val
                    m_test = l_val + d_test
                    a_test = m_test - c_val
                    
                    # Check validity
                    if (a_test >= par.grid_a_pd[0] and a_test <= par.grid_a_pd[-1] and
                        n_test >= 0 and n_test >= par.grid_n[0] and m_test >= par.grid_m[0]):
                        # Interpolate w(n_test, a_test)
                        w_val = linear_interp.interp_2d(
                            par.grid_b_pd, par.grid_a_pd, w, n_test, a_test
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
            
            if (a_test >= par.grid_a_pd[0] and a_test <= par.grid_a_pd[-1] and
                n_test >= 0 and n_test >= par.grid_n[0] and m_test >= par.grid_m[0]):
                w_val = linear_interp.interp_2d(
                    par.grid_b_pd, par.grid_a_pd, w, n_test, a_test
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
    
    c_out = sol.c[t]
    d_out = sol.d[t]
    v_out = np.zeros((par.Nn, par.Nm))
    
    upperenvelope_2d_segm(
        n_endo_2d, m_endo_2d, c_endo_2d, d_endo_2d, v_endo_2d,
        par.grid_n, par.grid_m, c_out, d_out, v_out
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
