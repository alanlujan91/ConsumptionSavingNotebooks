"""NEGMSmooth - NEGM with taste shock smoothing

This module extends the standard NEGM algorithm with taste shock smoothing
over discrete choice alternatives. It imports the pure consumption solver from
NEGM and only modifies the outer loop aggregation step.

The key difference: instead of taking the discrete max over alternatives
(optimal d, d=0 constraint, corner solution), we use a smooth max (logsum) 
with extreme value type I (Gumbel) taste shocks.
"""

import numpy as np
from numba import njit

# Import pure consumption solver from standard NEGM
from NEGM import solve_pure_c, obj_outer

# consav
from consav import linear_interp
from consav import golden_section_search

# local modules
import utility
import pens


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
def solve_outer(t,sol,par):
    """
    Solve outer problem using NEGM with optional taste shock smoothing.
    
    This function is identical to NEGM.solve_outer except for the aggregation step.
    When par.sigma > 0, it uses a smooth max (logsum) instead of discrete max.
    """

    # unpack output
    inv_v = sol.inv_v[t]
    inv_vm = sol.inv_vm[t]
    c = sol.c[t]
    d = sol.d[t]

    # loop over outer states
    for i_n in range(par.Nn):
            
        n = par.grid_n[i_n]

        # loop over m state
        for i_m in range(par.Nm):
            
            m = par.grid_m[i_m]
            
            # a. compute optimal choice (interior solution)
            d_low = 1e-8
            d_high = m-1e-8
            d_opt = golden_section_search.optimizer(obj_outer,d_low,d_high,args=(n,m,t,sol,par),tol=1e-8)

            # b. value and policy for interior solution
            n_opt = n + d_opt + pens.func(d_opt,par)
            m_opt = m - d_opt
            c_opt = np.fmin(linear_interp.interp_2d(par.grid_b_pd,par.grid_l,sol.c_pure_c[t],n_opt,m_opt),m_opt)
            v_opt = -obj_outer(d_opt,n,m,t,sol,par)

            # c. value and policy for d=0 constraint (dcon)
            v_dcon = -obj_outer(0,n,m,t,sol,par)
            c_dcon = linear_interp.interp_2d(par.grid_b_pd,par.grid_l,sol.c_pure_c[t],n,m)
            d_dcon = 0.0

            # d. value and policy for corner solution (con)
            w = linear_interp.interp_2d(par.grid_b_pd,par.grid_a_pd,sol.w[t],n,0)
            v_con = -1.0/(utility.func(m,par) + w)
            c_con = m
            d_con = 0.0

            # e. DIFFERENCE FROM STANDARD NEGM: smooth aggregation when sigma > 0
            if par.sigma > 0.0:
                # Collect alternatives
                alt_values = np.array([v_opt, v_dcon, v_con])
                alt_c = np.array([c_opt, c_dcon, c_con])
                alt_d = np.array([d_opt, d_dcon, d_con])
                
                # Filter valid alternatives (non-NaN, non-inf)
                n_valid = 0
                valid_values = np.zeros(3)
                valid_c = np.zeros(3)
                valid_d = np.zeros(3)
                
                for j in range(3):
                    if not np.isnan(alt_values[j]) and not np.isinf(alt_values[j]):
                        valid_values[n_valid] = alt_values[j]
                        valid_c[n_valid] = alt_c[j]
                        valid_d[n_valid] = alt_d[j]
                        n_valid += 1
                
                if n_valid == 0:
                    # No valid alternatives - should not happen
                    inv_v[i_n,i_m] = np.nan
                    c[i_n,i_m] = np.nan
                    d[i_n,i_m] = np.nan
                elif n_valid == 1:
                    # Only one valid alternative
                    inv_v[i_n,i_m] = valid_values[0]
                    c[i_n,i_m] = valid_c[0]
                    d[i_n,i_m] = valid_d[0]
                else:
                    # Multiple valid alternatives - use smooth max
                    smooth_v = logsum(valid_values[:n_valid], par.sigma)
                    inv_v[i_n,i_m] = smooth_v
                    
                    # Compute probability-weighted policies
                    probs = choice_prob(valid_values[:n_valid], par.sigma)
                    c[i_n,i_m] = np.sum(probs * valid_c[:n_valid])
                    d[i_n,i_m] = np.sum(probs * valid_d[:n_valid])
            else:
                # Standard discrete max (sigma = 0, same as standard NEGM)
                inv_v[i_n,i_m] = v_opt
                c[i_n,i_m] = c_opt
                d[i_n,i_m] = d_opt
                
                # Check if dcon is better
                if v_dcon > inv_v[i_n,i_m]:
                    c[i_n,i_m] = c_dcon
                    d[i_n,i_m] = d_dcon
                    inv_v[i_n,i_m] = v_dcon

                # Check if con is better
                if v_con > inv_v[i_n,i_m]:
                    c[i_n,i_m] = c_con
                    d[i_n,i_m] = d_con
                    inv_v[i_n,i_m] = v_con

            # f. derivative (IDENTICAL to standard NEGM)
            inv_vm[i_n,i_m] = 1.0/utility.marg_func(c[i_n,i_m],par)
