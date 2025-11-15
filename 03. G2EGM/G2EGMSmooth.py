"""G2EGMSmooth - G2EGM with taste shock smoothing

This module extends the standard G2EGM algorithm with taste shock smoothing
over discrete choice alternatives. It imports all segment solution methods from
G2EGM and only modifies the final aggregation step.

The key difference: instead of taking the discrete max over segment values,
we use a smooth max (logsum) with extreme value type I (Gumbel) taste shocks.
"""

import numpy as np
from numba import njit

# Import all segment solution functions from standard G2EGM
from G2EGM import solve_ucon, solve_dcon, solve_acon, solve_con

# consav
from consav import linear_interp # for linear interpolation

# local modules
import pens
import utility

@njit
def logsum(v_array, sigma):
    """
    Compute the smooth max using the logsum formula.
    
    For taste shocks with extreme value type I distribution with scale parameter sigma:
    E[max(v_j + epsilon_j)] = sigma * log(sum(exp(v_j / sigma)))
    
    Args:
        v_array: array of values for each alternative
        sigma: scale parameter for taste shocks (larger = more smoothing)
    
    Returns:
        smooth maximum value
    """
    if sigma <= 0.0:
        # If sigma is 0 or negative, return standard max
        return np.max(v_array)
    
    # Subtract max for numerical stability
    v_max = np.max(v_array)
    v_shifted = v_array - v_max
    
    # Compute logsum
    log_sum_exp = np.log(np.sum(np.exp(v_shifted / sigma)))
    
    return v_max + sigma * log_sum_exp

@njit
def choice_prob(v_array, sigma):
    """
    Compute choice probabilities using the logit formula.
    
    Args:
        v_array: array of values for each alternative
        sigma: scale parameter for taste shocks
    
    Returns:
        array of choice probabilities
    """
    if sigma <= 0.0:
        # If sigma is 0 or negative, return indicator for max
        probs = np.zeros_like(v_array)
        probs[np.argmax(v_array)] = 1.0
        return probs
    
    # Subtract max for numerical stability
    v_max = np.max(v_array)
    v_shifted = v_array - v_max
    
    # Compute probabilities
    exp_v = np.exp(v_shifted / sigma)
    probs = exp_v / np.sum(exp_v)
    
    return probs

@njit
def solve(t,sol,par):
    """
    Solve using G2EGM with optional taste shock smoothing.
    
    This function is identical to G2EGM.solve except for the aggregation step (section b).
    When par.sigma > 0, it uses a smooth max (logsum) instead of discrete argmax.
    """

    w = sol.w[t]
    wa = sol.wa[t]
    wb = sol.wb[t]

    # a. solve each segment (IDENTICAL to standard G2EGM)
    solve_ucon(sol.ucon_c[t,:,:],sol.ucon_d[t,:,:],sol.ucon_v[t,:,:],w,wa,wb,par)
    solve_dcon(sol.dcon_c[t,:,:],sol.dcon_d[t,:,:],sol.dcon_v[t,:,:],w,wa,par)
    solve_acon(sol.acon_c[t,:,:],sol.acon_d[t,:,:],sol.acon_v[t,:,:],w,wb,par)
    solve_con(sol.con_c[t,:,:],sol.con_d[t,:,:],sol.con_v[t,:,:],w,par)

    # b. upper envelope - ONLY DIFFERENCE: smooth aggregation
    seg_values = np.zeros(4)
    seg_c = np.zeros(4)
    seg_d = np.zeros(4)
    
    for i_n in range(par.Nn):
        for i_m in range(par.Nm):

            # i. collect segment values and policies
            seg_values[0] = sol.ucon_v[t,i_n,i_m]
            seg_values[1] = sol.dcon_v[t,i_n,i_m]
            seg_values[2] = sol.acon_v[t,i_n,i_m]
            seg_values[3] = sol.con_v[t,i_n,i_m]
            
            seg_c[0] = sol.ucon_c[t,i_n,i_m]
            seg_c[1] = sol.dcon_c[t,i_n,i_m]
            seg_c[2] = sol.acon_c[t,i_n,i_m]
            seg_c[3] = sol.con_c[t,i_n,i_m]
            
            seg_d[0] = sol.ucon_d[t,i_n,i_m]
            seg_d[1] = sol.dcon_d[t,i_n,i_m]
            seg_d[2] = sol.acon_d[t,i_n,i_m]
            seg_d[3] = sol.con_d[t,i_n,i_m]

            # ii. identify valid alternatives (non-NaN, non-inf values)
            n_valid = 0
            valid_values = np.zeros(4)
            valid_idx = np.zeros(4, dtype=np.int64)
            
            for i in range(4):
                if not np.isnan(seg_values[i]) and not np.isinf(seg_values[i]):
                    valid_values[n_valid] = seg_values[i]
                    valid_idx[n_valid] = i
                    n_valid += 1
            
            if n_valid == 0:
                # No valid alternatives - should not happen
                sol.inv_v[t,i_n,i_m] = np.nan
                sol.c[t,i_n,i_m] = np.nan
                sol.d[t,i_n,i_m] = np.nan
                continue
            elif n_valid == 1:
                # Only one valid alternative - use it directly
                idx = valid_idx[0]
                sol.inv_v[t,i_n,i_m] = -1.0/seg_values[idx]
                sol.c[t,i_n,i_m] = seg_c[idx]
                sol.d[t,i_n,i_m] = seg_d[idx]
            else:
                # Multiple valid alternatives - use smooth max
                # iii. compute smooth value using logsum on valid values only
                smooth_v = logsum(valid_values[:n_valid], par.sigma)
                sol.inv_v[t,i_n,i_m] = -1.0/smooth_v

                # iv. compute choice probabilities for valid alternatives
                valid_probs = choice_prob(valid_values[:n_valid], par.sigma)
                
                # v. compute expected optimal choices (probability-weighted)
                c_expected = 0.0
                d_expected = 0.0
                for j in range(n_valid):
                    idx = valid_idx[j]
                    c_expected += valid_probs[j] * seg_c[idx]
                    d_expected += valid_probs[j] * seg_d[idx]
                
                sol.c[t,i_n,i_m] = c_expected
                sol.d[t,i_n,i_m] = d_expected
        
    # c. derivatives 
    
    # i. m
    vm = utility.marg_func(sol.c[t],par)
    sol.inv_vm[t,:,:] = 1.0/vm

    # ii. n         
    a = par.grid_m_nd - sol.c[t] - sol.d[t]
    b = par.grid_n_nd + sol.d[t] + pens.func(sol.d[t],par)

    wb_now = np.zeros(a.shape)
    linear_interp.interp_2d_vec(par.grid_b_pd,par.grid_a_pd,wb,b.ravel(),a.ravel(),wb_now.ravel())
    
    vn = wb_now
    sol.inv_vn[t,:,:] = 1.0/vn
