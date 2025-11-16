"""Manual SEGM solver - Pure Python, no numba - for algorithm verification"""

import numpy as np
from consav import upperenvelope
import utility

# Create 1D upper envelope for consumption subproblem (same as NEGM)
negm_upperenvelope = upperenvelope.create(utility.func, use_inv_w=False)

def upperenvelope_2d_segm(n_endo_2d, m_endo_2d, c_endo_2d, d_endo_2d, v_endo_2d,
                          grid_n, grid_m, c_out, d_out, v_out, par):
    """
    2D upper envelope for SEGM - proper triangulation and barycentric interpolation
    
    Unlike G2EGM's upper envelope which chooses over both c and d,
    this only chooses over d (c is already determined from subproblem 1)
    
    Follows same structure as upperenvelope.py but adapted for SEGM:
    - Forms triangles from (b, l) structured grid
    - Maps to (n, m) space
    - Uses barycentric interpolation
    - Selects upper envelope
    - Fills holes
    
    Args:
        n_endo_2d, m_endo_2d: endogenous states [Nb, Nl]
        c_endo_2d, d_endo_2d, v_endo_2d: policies and values [Nb, Nl]
        grid_n, grid_m: target exogenous grids
        c_out, d_out, v_out: output arrays [Nn, Nm]
        par: parameters (for grid info)
    """
    Nb, Nl = n_endo_2d.shape
    Nn = len(grid_n)
    Nm = len(grid_m)
    
    # Initialize outputs
    c_out[:, :] = np.nan
    d_out[:, :] = np.nan
    v_out[:, :] = -np.inf
    
    # Mark valid points
    valid = np.ones((Nb, Nl), dtype=bool)
    for i_b in range(Nb):
        for i_l in range(Nl):
            valid[i_b, i_l] &= ~np.isnan(v_endo_2d[i_b, i_l])
            valid[i_b, i_l] &= ~np.isinf(v_endo_2d[i_b, i_l])
            valid[i_b, i_l] &= c_endo_2d[i_b, i_l] > 1e-8
            valid[i_b, i_l] &= m_endo_2d[i_b, i_l] > -0.1
            valid[i_b, i_l] &= n_endo_2d[i_b, i_l] > -0.1
    
    if valid.sum() < 100:
        return
    
    # Form triangles and interpolate (following upperenvelope.py structure)
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

def _process_triangle(i_l, i_b, tri, m, n, c, d, v, Nl, Nb, valid,
                     grid_n, grid_m, c_out, d_out, v_out):
    """Process one triangle for upper envelope (adapted from upperenvelope.py)"""
    
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
    m_min, m_max = min(m1, m2, m3), max(m1, m2, m3)
    n_min, n_max = min(n1, n2, n3), max(n1, n2, n3)
    
    # Find grid indices
    im_low = np.searchsorted(grid_m, m_min, side='left')
    im_high = np.searchsorted(grid_m, m_max, side='right')
    in_low = np.searchsorted(grid_n, n_min, side='left')
    in_high = np.searchsorted(grid_n, n_max, side='right')
    
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

def _fill_holes(c_out, d_out, v_out, grid_n, grid_m):
    """Fill holes (grid points not covered by triangulation) using nearest neighbor"""
    Nn, Nm = c_out.shape
    
    for i_n in range(Nn):
        for i_m in range(Nm):
            if np.isnan(c_out[i_n, i_m]) or np.isinf(v_out[i_n, i_m]):
                # Find nearest valid neighbor
                best_dist = np.inf
                best_c, best_d, best_v = grid_m[i_m], 0.0, -1e10
                
                for j_n in range(max(0, i_n-3), min(Nn, i_n+4)):
                    for j_m in range(max(0, i_m-3), min(Nm, i_m+4)):
                        if not np.isnan(c_out[j_n, j_m]) and not np.isinf(v_out[j_n, j_m]):
                            dist = (grid_n[i_n] - grid_n[j_n])**2 + (grid_m[i_m] - grid_m[j_m])**2
                            if dist < best_dist:
                                best_dist = dist
                                best_c = c_out[j_n, j_m]
                                best_d = d_out[j_n, j_m]
                                best_v = v_out[j_n, j_m]
                
                c_out[i_n, i_m] = best_c
                d_out[i_n, i_m] = best_d
                v_out[i_n, i_m] = best_v

def solve_segm_manual(w, wa, wb, par, model):
    """
    Manual SEGM solver for one period
    
    Inputs:
        w: post-decision value function w(b, a) on (grid_b_pd, grid_a_pd)
        wa: marginal value dw/da on (grid_b_pd, grid_a_pd)
        wb: marginal value dw/db on (grid_b_pd, grid_a_pd)
        par: parameters
    
    Outputs:
        c, d, v on (grid_n, grid_m)
    """
    
    grid_n = par.grid_n
    grid_m = par.grid_m
    grid_a_pd = par.grid_a_pd
    grid_b_pd = par.grid_b_pd
    
    Ra = par.Ra
    Rb = par.Rb
    chi = par.chi
    beta = par.beta
    rho = par.rho
    
    # Utility functions
    def u_prime(c):
        return c**(-rho)
    
    def inv_u_prime(x):
        return x**(-1/rho)
    
    def u(c):
        return c**(1-rho) / (1-rho)
    
    def psi_d(d):
        return chi * np.log(1 + d)
    
    def interp_2d(x, y, z, x_val, y_val):
        """Simple bilinear interpolation"""
        from consav import linear_interp
        return linear_interp.interp_2d(x, y, z, x_val, y_val)
    
    print(f"\n{'='*70}")
    print("MANUAL SEGM SOLVER")
    print(f"{'='*70}")
    print(f"Parameters: Ra={Ra:.4f}, Rb={Rb:.4f}, chi={chi:.4f}, beta={beta:.4f}, rho={rho:.4f}")
    print(f"Grids: n={len(grid_n)}, m={len(grid_m)}, a_pd={len(grid_a_pd)}, b_pd={len(grid_b_pd)}")
    print(f"Post-decision w: [{w.min():.4f}, {w.max():.4f}]")
    print(f"Post-decision wa: [{wa.min():.4f}, {wa.max():.4f}]")
    print(f"Post-decision wb: [{wb.min():.4f}, {wb.max():.4f}]")
    
    # =========================================================================
    # SUBPROBLEM 1: Pure Consumption (b, a) → (b, l)
    # =========================================================================
    print(f"\nSubproblem 1: Pure Consumption...")
    
    # Build endogenous grid
    c_endo_list = []
    l_endo_list = []
    b_endo_list = []
    w_endo_list = []
    
    for i_b in range(len(grid_b_pd)):
        for i_a in range(len(grid_a_pd)):
            b_val = grid_b_pd[i_b]
            a_val = grid_a_pd[i_a]
            wa_val = wa[i_b, i_a]
            
            if wa_val > 1e-10:
                # Euler equation: u'(c) = β * Ra * wa
                c_val = inv_u_prime(beta * Ra * wa_val)
                l_val = a_val + c_val  # Liquid resources
                
                c_endo_list.append(c_val)
                l_endo_list.append(l_val)
                b_endo_list.append(b_val)
                w_endo_list.append(w[i_b, i_a])
    
    c_endo = np.array(c_endo_list)
    l_endo = np.array(l_endo_list)
    b_endo = np.array(b_endo_list)
    w_endo = np.array(w_endo_list)
    
    print(f"  Endogenous points: {len(c_endo)}")
    print(f"  l_endo: [{l_endo.min():.4f}, {l_endo.max():.4f}]")
    print(f"  c_endo: [{c_endo.min():.4f}, {c_endo.max():.4f}]")
    
    # Regrid to (b×, l×) where l× = grid_m using 1D upper envelope for each b
    c_pure_c = np.zeros((len(grid_b_pd), len(grid_m)))
    v_pure_c = np.zeros((len(grid_b_pd), len(grid_m)))
    v_l = np.zeros((len(grid_b_pd), len(grid_m)))
    v_b = np.zeros((len(grid_b_pd), len(grid_m)))
    
    # Use jit model for numba compatibility for BOTH subproblems
    from EconModel import jit
    with jit(model) as jit_model:
        jit_par = jit_model.par
        
        # =====================================================================
        # SUBPROBLEM 1: Regrid using upper envelope
        # =====================================================================
        for i_b in range(len(grid_b_pd)):
            b_val = grid_b_pd[i_b]
            
            # Prepare temporary arrays for this b slice
            temp_c = np.zeros(len(grid_a_pd))
            temp_l = np.zeros(len(grid_a_pd))
            temp_v = np.zeros(len(grid_m))
            
            # Build endogenous grid from unconstrained Euler equation
            # NOTE: wa already includes β * Ra from post_decision.py
            for i_a in range(len(grid_a_pd)):
                a_val = grid_a_pd[i_a]
                wa_val = wa[i_b, i_a]
                
                # Use same approach as NEGM
                temp_c[i_a] = inv_u_prime(wa_val)
                temp_l[i_a] = a_val + temp_c[i_a]  # endogenous l
            
            # Call 1D upper envelope (same as NEGM)
            # The upper envelope handles constraint-binding points internally
            negm_upperenvelope(grid_a_pd, temp_l, temp_c, w[i_b],
                               grid_m, c_pure_c[i_b, :], temp_v, jit_par)
            
            # Compute marginal values using envelope theorem
            for i_l in range(len(grid_m)):
                l_val = grid_m[i_l]
                c_val = c_pure_c[i_b, i_l]
                a_val = l_val - c_val
                
                # v_l = u'(c)
                v_l[i_b, i_l] = u_prime(c_val) if c_val > 1e-10 else 1e10
                
                # v_b = wb(b, a)
                if a_val >= grid_a_pd[0] and a_val <= grid_a_pd[-1]:
                    v_b[i_b, i_l] = interp_2d(grid_b_pd, grid_a_pd, wb, b_val, a_val)
                else:
                    v_b[i_b, i_l] = 0.0
                
                # Value from upper envelope
                v_pure_c[i_b, i_l] = temp_v[i_l]
        
        print(f"  c_pure_c: [{c_pure_c.min():.4f}, {c_pure_c.max():.4f}]")
        print(f"  v_l: [{v_l.min():.4f}, {v_l.max():.4f}]")
        print(f"  v_b: [{v_b.min():.4f}, {v_b.max():.4f}]")
        
        # =====================================================================
        # SUBPROBLEM 2: Pure Pension (b, l) → (n, m)
        # =====================================================================
        print(f"\nSubproblem 2: Pure Pension...")
        
        # Build endogenous grid in 2D structure for upper envelope
        Nb = len(grid_b_pd)
        Nl = len(grid_m)
        
        d_endo_2d = np.full((Nb, Nl), np.nan)
        c_endo_2d = np.full((Nb, Nl), np.nan)
        n_endo_2d = np.full((Nb, Nl), np.nan)
        m_endo_2d = np.full((Nb, Nl), np.nan)
        v_endo_2d = np.full((Nb, Nl), -np.inf)
        
        for i_b in range(Nb):
            b_val = grid_b_pd[i_b]
            
            for i_l in range(Nl):
                l_val = grid_m[i_l]
                
                c_val = c_pure_c[i_b, i_l]
                v_l_val = v_l[i_b, i_l]
                v_b_val = v_b[i_b, i_l]
                
                # Skip if c is invalid
                if c_val <= 1e-8:
                    continue
                
                # FOC for d: ψ'(d) = v_l - v_b
                # chi/(1+d) = v_l - v_b
                # d = chi/(v_l - v_b) - 1
                
                denom = v_l_val - v_b_val
                
                # Try unconstrained solution first
                d_unconstrained = np.nan
                v_unconstrained = -np.inf
                
                if denom > 1e-6 and v_b_val > 1e-10 and v_l_val < 1e9:
                    d_test = chi / denom - 1
                    
                    if d_test > 0:
                        d_test = min(d_test, b_val, l_val)
                        
                        # Compute endogenous (n, m)
                        psi_val = psi_d(d_test)
                        n_test = b_val - d_test - psi_val
                        m_test = l_val + d_test
                        a_test = m_test - c_val
                        
                        if (a_test >= grid_a_pd[0] and a_test <= grid_a_pd[-1] and 
                            n_test >= 0 and n_test >= grid_n[0] and m_test >= grid_m[0]):
                            w_val = interp_2d(grid_b_pd, grid_a_pd, w, n_test, a_test)
                            v_test = u(c_val) + psi_d(d_test) + w_val
                            
                            d_unconstrained = d_test
                            v_unconstrained = v_test
                
                # Always try d=0 constraint-binding point
                d_test = 0.0
                psi_val = psi_d(d_test)
                n_test = b_val - d_test - psi_val
                m_test = l_val + d_test
                a_test = m_test - c_val
                
                v_constrained = -np.inf
                if (a_test >= grid_a_pd[0] and a_test <= grid_a_pd[-1] and 
                    n_test >= 0 and n_test >= grid_n[0] and m_test >= grid_m[0]):
                    w_val = interp_2d(grid_b_pd, grid_a_pd, w, n_test, a_test)
                    v_constrained = u(c_val) + psi_d(d_test) + w_val
                
                # Pick the best solution (upper envelope will select globally later)
                if v_unconstrained > v_constrained:
                    d_endo_2d[i_b, i_l] = d_unconstrained
                    psi_val = psi_d(d_unconstrained)
                    n_endo_2d[i_b, i_l] = b_val - d_unconstrained - psi_val
                    m_endo_2d[i_b, i_l] = l_val + d_unconstrained
                    c_endo_2d[i_b, i_l] = c_val
                    v_endo_2d[i_b, i_l] = v_unconstrained
                elif v_constrained > -np.inf:
                    d_endo_2d[i_b, i_l] = 0.0
                    n_endo_2d[i_b, i_l] = b_val
                    m_endo_2d[i_b, i_l] = l_val
                    c_endo_2d[i_b, i_l] = c_val
                    v_endo_2d[i_b, i_l] = v_constrained
        
        # Count valid points
        valid_mask = ~np.isnan(d_endo_2d) & ~np.isinf(v_endo_2d)
        n_valid = np.sum(valid_mask)
        
        print(f"  Valid endogenous points: {n_valid} / {Nb * Nl}")
        if n_valid > 0:
            d_valid = d_endo_2d[valid_mask]
            n_valid_arr = n_endo_2d[valid_mask]
            m_valid_arr = m_endo_2d[valid_mask]
            v_valid = v_endo_2d[valid_mask]
            
            print(f"  d_endo: [{d_valid.min():.4f}, {d_valid.max():.4f}], mean={d_valid.mean():.3f}")
            print(f"  d_endo>0.01: {np.sum(d_valid > 0.01)} / {n_valid}")
            print(f"  n_endo: [{n_valid_arr.min():.4f}, {n_valid_arr.max():.4f}]")
            print(f"  m_endo: [{m_valid_arr.min():.4f}, {m_valid_arr.max():.4f}]")
            print(f"  v_endo: [{v_valid.min():.4f}, {v_valid.max():.4f}]")
        
        # Regrid using SEGM's custom 2D upper envelope with triangulation
        c_out = np.zeros((len(grid_n), len(grid_m)))
        d_out = np.zeros((len(grid_n), len(grid_m)))
        v_out = np.zeros((len(grid_n), len(grid_m)))
        
        if n_valid > 100:
            # Create par object for compatibility
            class ParDummy:
                pass
            par_dummy = ParDummy()
            
            # Use our custom SEGM upper envelope with proper triangulation
            upperenvelope_2d_segm(n_endo_2d, m_endo_2d, c_endo_2d, d_endo_2d, v_endo_2d,
                                 grid_n, grid_m, c_out, d_out, v_out, par_dummy)
    
    print(f"\n✓ Manual SEGM complete")
    print(f"  c: [{c_out.min():.4f}, {c_out.max():.4f}], mean={c_out.mean():.3f}")
    print(f"  d: [{d_out.min():.4f}, {d_out.max():.4f}], mean={d_out.mean():.3f}")
    print(f"  d>0.01: {np.sum(d_out > 0.01)} / {d_out.size}")
    print(f"  v: [{v_out.min():.4f}, {v_out.max():.4f}]")
    
    return c_out, d_out, v_out

