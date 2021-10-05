#############################################################################
# The Adaptive Lasso and Adaptive Group Lasso with stability selection and error control
# Author: Georges Tod 
#############################################################################

import numpy as np
import matplotlib.pylab as plt

from scipy.linalg import block_diag
from scipy.stats import beta

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LassoCV, Ridge
from sklearn.utils import shuffle
from sklearn.utils.random import sample_without_replacement
from sklearn.utils import resample

from joblib import Parallel, delayed

import jax
from jax import random, numpy as jnp





#############################################################################
####### Computing the mask
#############################################################################

def get_mask_single(theta,dt,randomized:bool=True):
    # copying from jnp array on GPU to np array on CPU
    theta = np.array(theta,dtype=np.float64)
    dt = np.array(dt,dtype=np.float64)
    # normalizing dt and library time wise
    normed_dt = dt/np.linalg.norm(dt,axis=0,keepdims=True)
    normed_theta = theta/np.linalg.norm(theta,axis=0,keepdims=True)
    # computing mask
    mask,extras = Adaptive_Lasso_SS(normed_theta,normed_dt,randomized=randomized)
    mask = np.array(mask,dtype=np.uint8).reshape(-1,1)
    
    return mask,extras

#############################################################################
####### Adaptive Lasso with stability selection and error control
#############################################################################

def Adaptive_Lasso_SS(X,y,
                     randomized: bool=True,      # True: randomized adaptive Lasso, False: adaptive Lasso
                     alphas: np.ndarray = None,  # as in classic scikit-learn                  
                     nI: int = 20,               # number of resamples for the stability selection (B in the paper)
                     ratio: int = 2,             # size of resamples 
                     n_alphas: int = 10,         # as in classic scikit-learn  
                     n_cores: int=-1,            # by default stability selection will be performed in parallel using all cores
                     eps: int=3,                 # regularisation path length
                     d1: float = 1,              # parameter 1 for the beta distribution (only if randomized=True)
                     d2: float = 2,              # parameter 2 for the beta distribution (only if randomized=True)
                     seed: int = 42,             # magic seed
                     super_power: float = 2,     # adaptive Lasso's super power denoted gamma in the paper
                     efp:int = 3,                # expected false positives upper bound EVmax in the paper
                     piT:float = 0.9):           # probability of being selected threshold \pi_{thr} in the paper
    
    if alphas is None:
        # computing range of alphas
        alpha_max = np.sqrt(np.sum(((X.T) @ y) ** 2)).max()/X.shape[0]
        e_amax = np.log10(alpha_max).round()
        e_amin = e_amax - eps
        alphas    = np.logspace(e_amin, e_amax, n_alphas, base=10)

    # some params
    n_population = X.shape[0]
    n_samples    = int(n_population/ratio)
    
    # randomizing
    np.random.seed(seed=seed)        
    if randomized:
        W = np.random.beta(d1,d2,size=[nI,X.shape[1]])
        W[W>1]   = 1
        W[W<0.1] = 0.1
        
    else:
        W = np.ones([nI,X.shape[1]])

    def stab_(i, alpha):  
        tau = np.zeros(X.shape[1])
        for j in range(nI):
            idx_ = sample_without_replacement(n_population=n_population,n_samples=n_samples)
            y_train = y[idx_]
            X_train = X[idx_,:] * W[j,:]
            ##########################################
            def adaLasso(alpha):
                n_samples, n_features = X.shape
                weights = (np.abs(Ridge(alpha=1e-10,fit_intercept=False).fit(X, y).coef_).ravel())**super_power
                X_w = X_train * weights[np.newaxis, :]
                clf = Lasso(alpha=alpha, fit_intercept=False,tol=1e-5)
                clf.fit(X_w, y_train)
                coef_ = clf.coef_ * weights
                return coef_
            ##########################################
            betahat = adaLasso(alpha) * W[j,:]
            active = (np.abs(betahat) > 0) *1/nI
            
            tau = tau + active
            
        return tau

    # selection probabilities
    pSE = Parallel(n_jobs=n_cores)(delayed(stab_)(i, alpha) for i, alpha in enumerate(alphas))
    tau = np.array(pSE)
        
        
    # average of selected variables
    q_hat = tau.sum(axis=1)
    # verifying on some upper bound on efp
    ev_region = (q_hat)**2/((2*piT-1)*X.shape[1])    
    idxS = (ev_region<efp).argmax()   
    minLambdaSS = alphas[idxS]
    # selecting variables where the efp is respected
    active_set = (tau[idxS:,:]>piT).any(axis=0)   
        
    mask = active_set
    mask = np.array(mask,dtype=np.uint8)


    for_plots = [tau,piT,pSE,alphas, minLambdaSS, active_set,(ev_region<efp)]
    
    return mask, for_plots

#############################################################################
####### tools
#############################################################################

def ir_cond(theta,GT):
    # copying from jnp array on GPU to np array on CPU
    theta = np.array(theta,dtype=np.float64)
    # normalizing
    normed_theta = theta/np.linalg.norm(theta,axis=0,keepdims=True)
    x1 = normed_theta[:,GT]
    x2 = normed_theta[:,np.setdiff1d(np.arange(0,theta.shape[1]),np.array(GT))]
    metric = np.abs(np.linalg.inv(x1.T @ x1) @ x1.T @ x2)
    irc_holds = np.all(metric<1) 
    cond_num = np.linalg.cond(normed_theta[:,:])
    return irc_holds, metric, cond_num

    
def ir_condAL(theta,dt,GT):
    # copying from jnp array on GPU to np array on CPU
    theta = np.array(theta,dtype=np.float64)
    dt = np.array(dt,dtype=np.float64)
    # normalizing
    normed_theta = theta/np.linalg.norm(theta,axis=0,keepdims=True)
    normed_dt = dt/np.linalg.norm(dt,axis=0,keepdims=True)
    weights = (np.abs(Ridge(alpha=1e-10,fit_intercept=False).fit(normed_theta,normed_dt).coef_).ravel())**2
    normed_theta_w = normed_theta * weights[np.newaxis, :]
    x1 = normed_theta_w[:,GT]
    x2 = normed_theta_w[:,np.setdiff1d(np.arange(0,theta.shape[1]),np.array(GT))]
    metric = np.abs(np.linalg.inv(x1.T @ x1) @ x1.T @ x2)
    irc_holds = np.all(metric<1) 
    cond_num = np.linalg.cond(normed_theta_w[:,:])

    return irc_holds, metric, cond_num