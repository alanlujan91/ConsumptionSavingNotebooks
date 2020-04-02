# -*- coding: utf-8 -*-
"""GEModelClass

Solves and simulates a buffer-stock consumption-saving problem for use in a general equilibrium model

"""

##############
# 1. imports #
##############

import time
import numpy as np
from numba import njit, prange

# consav
from consav import ModelClass
from consav import linear_interp
from consav.misc import elapsed, equilogspace, markov_rouwenhorst

############
# 2. model #
############

class GEModelClass(ModelClass):
    
    #########
    # setup #
    #########      

    def setup(self):
        """ set baseline parameters """

        par = self.par
        
        # a. define list of non-float scalars
        self.not_float_list = ['Ne','Na','max_iter_solve','max_iter_simulate','tp_T']
        
        # b. steady state values
        par.r_ss = np.nan
        par.w_ss = np.nan
        par.Y_ss = np.nan
        par.C_ss = np.nan
        par.kd_ss = np.nan
        par.ks_ss = np.nan

        # c. preferences
        par.sigma = 1.0 # CRRA coefficient
        par.beta = 0.982 # discount factor

        # d. production
        par.Z = 1.0 # technology level
        par.alpha = 0.11 # Cobb-Douglas coefficient
        par.delta = 0.025 # depreciation rate

        # e. income parameters
        par.rho = 0.966 # AR(1) parameter
        par.sigma_e = 0.50 # std. of persistent shock
        par.Ne = 7 # number of states

        # f. grids         
        par.a_max = 200.0 # maximum point in grid for a
        par.Na = 500 # number of grid points

        # g. misc.
        par.tp_T = 500 # length of transition path
        par.max_iter_solve = 5000 # maximum number of iterations when solving
        par.max_iter_simulate = 5000 # maximum number of iterations when simulating
        par.solve_tol = 1e-10 # tolerance when solving
        par.simulate_tol = 1e-10 # tolerance when simulating

    def allocate(self):
        """ allocate model, i.e. create grids and allocate solution and simluation arrays """

        par = self.par
        sol = self.sol
        sim = self.sim

        # a. grids
        self.create_grids()

        # a. solution
        sol_shape = (par.Ne,par.Na)
        sol.a = np.zeros(sol_shape)
        sol.m = np.zeros(sol_shape)
        sol.c = np.zeros(sol_shape)
        sol.Va = np.zeros(sol_shape)

        # transition path
        tp_sol_shape = (par.tp_T,par.Ne,par.Na)
        sol.tp_a = np.zeros(tp_sol_shape)
        sol.tp_m = np.zeros(tp_sol_shape)
        sol.tp_c = np.zeros(tp_sol_shape)
        sol.tp_Va = np.zeros(tp_sol_shape)

        # b. simulation
        sim_shape = sol_shape
        sim.i = np.zeros(sim_shape,dtype=np.int64)
        sim.w = np.zeros(sim_shape)
        sim.D = np.zeros(sim_shape)

        tp_sim_shape = tp_sol_shape
        sim.tp_i = np.zeros(tp_sim_shape,dtype=np.int64)
        sim.tp_w = np.zeros(tp_sim_shape)
        sim.tp_D = np.zeros(tp_sim_shape)

    def create_grids(self):
        """ construct grids for states and shocks """

        par = self.par

        par.a_grid = equilogspace(0,par.a_max,par.Na)
        par.e_grid, par.e_trans, par.e_ergodic = markov_rouwenhorst(par.rho,par.sigma_e,par.Ne)

    #########
    # solve #
    #########
    
    def implied_r(self,k,Z=None):
        """ implied r given k = K/L and optimal firm behavior """

        par = self.par
        if Z is None: Z = par.Z
        r = Z*par.alpha*k**(par.alpha-1)-par.delta
        return r

    def implied_w(self,r,Z=None):
        """ implied w given r and optimal firm behavior """

        par = self.par
        if Z is None: Z = par.Z
        w = Z*(1.0-par.alpha)*((r+par.delta)/(Z*par.alpha))**(par.alpha/(par.alpha-1))
        return w

    def firm_demand(self,r,Z=None):
        """ firm demand for k = K/L given r and optimal firm behavior """

        par = self.par
        if Z is None: Z = par.Z
        k = ((r+par.delta)/(Z*par.alpha))**(1/(par.alpha-1))
        return k

    def firm_production(self,k,Z=None):
        """ firm production """

        par = self.par
        if Z is None: Z = par.Z
        return Z*k**par.alpha

    def steady_state(self,r_ss=None,do_print=True):
        """ computate steady state statistics """

        par = self.par
        sol = self.sol
        sim = self.sim

        if not r_ss is None: par.r_ss = r_ss

        # a. firm
        par.w_ss = self.implied_w(par.r_ss)
        par.kd_ss = self.firm_demand(par.r_ss)
        par.Y_ss = self.firm_production(par.kd_ss)

        # b. solve household problem
        self.solve_household_ss(par.r_ss,do_print=do_print)
        self.simulate_household_ss(do_print=do_print)

        # implied supply of capital and consumption
        par.ks_ss = np.sum(sim.D*sol.a)
        par.C_ss = np.sum(sim.D*sol.c)

        # c. equilibrium conditions
        if do_print:
            print('')
            print(f'r: {par.r_ss:.4f}')
            print(f'w: {par.w_ss:.4f}')
            print(f'Y: {par.Y_ss:.4f}')
            print(f'K/Y: {par.kd_ss/par.Y_ss:.4f}')
            print('')
            print(f'capital market clearing: {par.ks_ss-par.kd_ss:12.8f}')
            print(f'goods market clearing: {par.Y_ss-par.C_ss-par.delta*par.ks_ss:12.8f}')

    def solve_household_ss(self,r,Va=None,do_print=False):
        """ gateway for solving the model in steady state """

        par = self.par
        sol = self.sol
        t0 = time.time()
        
        # a. find wage from optimal firm behavior
        w = self.implied_w(r)

        # b. create (or re-create) grids
        self.create_grids()

        # c. solve
        sol.m = (1+r)*par.a_grid[np.newaxis,:] + w*par.e_grid[:,np.newaxis]
        sol.Va = (1+r)*(0.1*sol.m)**(-par.sigma) if Va is None else Va

        it = 0
        while True:

            # i. save
            a_old = sol.a.copy()

            # ii. egm
            time_iteration(par,r,w,sol.Va,sol.Va,sol.a,sol.c,sol.m)

            # ii. check
            if np.max(np.abs(sol.a-a_old)) < par.solve_tol: break
            
            # iv. increment
            it += 1
            if it > par.max_iter_solve: raise Exception('too many iterations when solving for steady state')

        if do_print:
            print(f'household problem solved in {elapsed(t0)} [{it} iterations]')

    def solve_household_tp(self,tp_r,do_print=False):
        """ gateway for solving the model along the transition path """

        par = self.par
        sol = self.sol
        t0 = time.time()

        # a. create (or re-create) grids
        self.create_grids()

        # c. solve
        for t in reversed(range(par.tp_T)):
            
            # i. prices
            r = tp_r[t]
            w = self.implied_w(tp_r[t])

            # ii. next-period
            if t == par.tp_T-1:
                Va_p = sol.Va
            else:
                Va_p = sol.tp_Va[t+1]

            # ii. solve
            sol.tp_m[t] = (1+r)*par.a_grid[np.newaxis,:] + w*par.e_grid[:,np.newaxis]

            # iii. time iteration
            time_iteration(par,r,w,Va_p,sol.tp_Va[t],sol.tp_a[t],sol.tp_c[t],sol.tp_m[t])

        if do_print:
            print(f'household problem solved in {elapsed(t0)}')

    def simulate_household_ss(self,D=None,do_print=False):
        """ gateway for simulating the model towards the steady state"""
        
        par = self.par
        sol = self.sol
        sim = self.sim        
        t0 = time.time()

        # a. intial guess
        D = (np.repeat(par.e_ergodic,par.Na)/par.Na).reshape(par.Ne,par.Na) if D is None else D

        # b. simulate
        it = simulate_ss(par,sol,sim,D)

        if do_print:
            print(f'household problem simulated in {elapsed(t0)} [{it} iterations]')

    def simulate_household_tp(self,do_print=False):
        """ gateway for simulating the model along the transition path"""
        
        par = self.par
        sol = self.sol
        sim = self.sim        
        t0 = time.time()

        simulate_path(par,sol,sim)

        if do_print:
            print(f'household problem simulated in {elapsed(t0)}')

######################
# fast jit functions #
######################

@njit(parallel=True)        
def time_iteration(par,r,w,Va_p,Va,a,c,m):
    """ perform time iteration step with Va_p from previous iteration """

    # a. post-decision 
    marg_u_plus = (par.beta*par.e_trans)@Va_p

    # b. egm loop
    for i_e in prange(par.Ne):
        
        # i. egm
        c_endo = marg_u_plus[i_e]**(-1/par.sigma)
        m_endo = c_endo + par.a_grid

        # ii. interpolation
        linear_interp.interp_1d_vec(m_endo,par.a_grid,m[i_e],a[i_e])
        a[i_e,0] = np.fmax(a[i_e,0],0)
        c[i_e] = m[i_e]-a[i_e]

        # iii. envelope condition
        Va[i_e] = (1+r)*c[i_e]**(-par.sigma)
    
@njit
def binary_search(imin,x,xi):
    """ binary search algorithm """

    Nx = x.size

    # a. checks
    if xi <= x[0]:
        return 0
    elif xi >= x[Nx-2]:
        return Nx-2
    
    # b. binary search
    half = Nx//2
    while half:
        imid = imin + half
        if x[imid] <= xi:
            imin = imid
        Nx -= half
        half = Nx//2
        
    return imin

@njit(parallel=True) 
def find_i_and_w(par,sim,a):
    """ find indices and weights for simulation """

    i = sim.i
    w = sim.w

    for i_e in prange(par.Ne):
        for i_a in prange(par.Na):
            
            # a. policy
            a_ = a[i_e,i_a]

            # b. find i_ such par.a_grid[i_] <= a_ < par.a_grid[i_+1]
            i_ = i[i_e,i_a] = binary_search(0,par.a_grid,a_) 

            # b. weight
            w[i_e,i_a] = (par.a_grid[i_+1] - a_) / (par.a_grid[i_+1] - par.a_grid[i_])

@njit(parallel=True)   
def simulate(par,sim,e_trans_T,D):
    """ simulate given weight indices are weight """

    # a. assuming e is constant
    Dnew = np.zeros(D.shape)
    for i_e in prange(par.Ne):
        for i_a in range(par.Na):
            
            # a. from
            D_ = D[i_e,i_a]
            w = sim.w[i_e,i_a]

            # b. to
            i = sim.i[i_e,i_a]            
            Dnew[i_e,i] += D_*w
            Dnew[i_e,i+1] += D_*(1.0-w)
    
    # b. account for transition probability of e
    Dnew_ = e_trans_T@Dnew

    return Dnew_

@njit(parallel=True)        
def simulate_ss(par,sol,sim,D):
    """ simulate forwards to steady state """

    # a. indices and weights
    find_i_and_w(par,sim,sol.a)

    # b. iterate
    e_trans_T = par.e_trans.T.copy()
    it = 0
    while True:

        # i. new distribution
        Dnew = simulate(par,sim,e_trans_T,D)

        # ii. check
        if np.max(np.abs(Dnew-D)) < par.simulate_tol: break
        D = Dnew

        # iii. increment
        it += 1
        if it > par.max_iter_simulate: raise Exception()

    # d. save
    sim.D = D
    return it
    
def simulate_path(par,sol,sim):
    """ simulate along transition path """

    e_trans_T = par.e_trans.T.copy()
    for t in range(par.tp_T):

        # a. indices and weights
        find_i_and_w(par,sim,sol.tp_a[t])

        # b. new distribution
        if t == 0:
            Dlag = sim.D
        else:
            Dlag = sim.tp_D[t-1]

        sim.tp_D[t] = simulate(par,sim,e_trans_T,Dlag)