import numpy as np
from numba import njit

# consav
from consav import linear_interp # for linear interpolation

@njit
def logsumexp(v1, v2, sigma):
    """
    Compute log(exp(v1/sigma) + exp(v2/sigma)) * sigma using numerically stable formula.
    
    When sigma → 0, this approaches max(v1, v2).
    When sigma → ∞, this approaches (v1 + v2)/2 + sigma*log(2).
    
    Args:
        v1, v2: values to compare (can be negative utility values)
        sigma: taste shock standard deviation (> 0)
    
    Returns:
        Smoothed maximum value
    """
    if sigma < 1e-10:
        # No taste shocks, return hard max
        return np.maximum(v1, v2)
    
    # Numerically stable logsumexp
    v_max = np.maximum(v1, v2)
    return v_max + sigma * np.log(np.exp((v1 - v_max) / sigma) + np.exp((v2 - v_max) / sigma))

@njit
def choice_prob(v_chosen, v_other, sigma):
    """
    Compute probability of choosing option with value v_chosen over v_other.
    
    Uses logit formula: P = exp(v_chosen/sigma) / (exp(v_chosen/sigma) + exp(v_other/sigma))
    
    When sigma → 0, this approaches 1 if v_chosen > v_other, 0.5 if equal, 0 otherwise.
    When sigma → ∞, this approaches 0.5 (random choice).
    
    Args:
        v_chosen: value of chosen option
        v_other: value of other option  
        sigma: taste shock standard deviation (> 0)
    
    Returns:
        Choice probability in [0, 1]
    """
    if sigma < 1e-10:
        # No taste shocks, return hard choice
        if v_chosen > v_other:
            return 1.0
        elif v_chosen < v_other:
            return 0.0
        else:
            return 0.5
    
    # Numerically stable logit
    diff = (v_chosen - v_other) / sigma
    
    # Avoid overflow: if diff is very large, probability ≈ 1
    if diff > 50:
        return 1.0
    elif diff < -50:
        return 0.0
    
    exp_diff = np.exp(diff)
    return exp_diff / (1.0 + exp_diff)

@njit
def compute(t,sol,par,G2EGM=True):

    # unpack
    w = sol.w[t]
    wa = sol.wa[t]
    if G2EGM:
        wb = sol.wb[t]

    # loop over outermost post-decision state
    for i_b in range(par.Nb_pd):

        # a. initialize
        w[i_b,:] = 0
        wa[i_b,:] = 0
        if G2EGM:
            wb[i_b,:] = 0

        # b. working memoery
        inv_v_plus = np.zeros(par.Na_pd)
        inv_vm_plus = np.zeros(par.Na_pd)
        if G2EGM:
            inv_vn_plus = np.zeros(par.Na_pd)
        
        inv_v_ret_plus = np.zeros(par.Na_pd)
        inv_vm_ret_plus = np.zeros(par.Na_pd)
        if G2EGM:
            inv_vn_ret_plus = np.zeros(par.Na_pd)

        # c. loop over shocks
        for i_eta in range(par.Neta):
            
            # i. next period states
            m_plus = par.Ra*par.grid_a_pd + par.eta[i_eta]
            n_plus = par.Rb*par.grid_b_pd[i_b]
            m_plus_ret = m_plus + n_plus

            # ii. prepare interpolation in p direction
            prep = linear_interp.interp_2d_prep(par.grid_n,n_plus,par.Na_pd)
            prep_ret = linear_interp.interp_1d_prep(par.Na_pd)

            # iii. interpolations

            # work
            linear_interp.interp_2d_only_last_vec_mon(prep,par.grid_n,par.grid_m,sol.inv_v[t+1],n_plus,m_plus,inv_v_plus)
            linear_interp.interp_2d_only_last_vec_mon_rep(prep,par.grid_n,par.grid_m,sol.inv_vm[t+1],n_plus,m_plus,inv_vm_plus)
            if G2EGM:
                linear_interp.interp_2d_only_last_vec_mon_rep(prep,par.grid_n,par.grid_m,sol.inv_vn[t+1],n_plus,m_plus,inv_vn_plus)

            # retire
            linear_interp.interp_1d_vec_mon(prep_ret,sol.m_ret[t+1],sol.inv_v_ret[t+1],m_plus_ret,inv_v_ret_plus)
            linear_interp.interp_1d_vec_mon_rep(prep_ret,sol.m_ret[t+1],sol.inv_vm_ret[t+1],m_plus_ret,inv_vm_ret_plus)
            if G2EGM:
                linear_interp.interp_1d_vec_mon_rep(prep_ret,sol.m_ret[t+1],sol.inv_vn_ret[t+1],m_plus_ret,inv_vn_ret_plus)

            # iv. accumulate
            for i_a in range(par.Na_pd):

                # Convert inverse values to values (inv_v = -1/v for negative v)
                v_work = -1.0 / inv_v_plus[i_a]
                v_ret = -1.0 / inv_v_ret_plus[i_a]
                
                # Use taste shocks if sigma > 0
                if par.sigma < 1e-10:
                    # No taste shocks: use hard max (original behavior)
                    if inv_v_ret_plus[i_a] > inv_v_plus[i_a]:
                        w_now = v_ret
                        wa_now = 1.0/inv_vm_ret_plus[i_a]
                        if G2EGM:
                            wb_now = 1.0/inv_vn_ret_plus[i_a]
                    else:
                        w_now = v_work
                        wa_now = 1.0/inv_vm_plus[i_a]
                        if G2EGM:
                            wb_now = 1.0/inv_vn_plus[i_a]
                else:
                    # Taste shocks: use smooth max and probability-weighted marginals
                    w_now = logsumexp(v_work, v_ret, par.sigma)
                    
                    # Compute choice probabilities
                    prob_work = choice_prob(v_work, v_ret, par.sigma)
                    prob_ret = 1.0 - prob_work
                    
                    # Weight marginal values by choice probabilities
                    wa_work = 1.0/inv_vm_plus[i_a]
                    wa_ret = 1.0/inv_vm_ret_plus[i_a]
                    wa_now = prob_work * wa_work + prob_ret * wa_ret
                    
                    if G2EGM:
                        wb_work = 1.0/inv_vn_plus[i_a]
                        wb_ret = 1.0/inv_vn_ret_plus[i_a]
                        wb_now = prob_work * wb_work + prob_ret * wb_ret
                
                w[i_b,i_a] += par.w_eta[i_eta]*par.beta*w_now
                wa[i_b,i_a] += par.w_eta[i_eta]*par.Ra*par.beta*wa_now
                if G2EGM:
                    wb[i_b,i_a] += par.w_eta[i_eta]*par.Rb*par.beta*wb_now