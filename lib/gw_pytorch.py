#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:06:02 2019

@author: vayer
"""

import torch

class NotImplementedError(Exception):
    pass 

def gromov_cost(xs,xt,T,device):
    C1 = _cost_matrix(xs,xs)
    C2 = _cost_matrix(xt,xt)
    p=torch.ones(xs.shape[0])/xs.shape[0]
    q=torch.ones(xt.shape[0])/xt.shape[0]
    p=p.to(device)
    q=q.to(device)  
    constC, hC1, hC2 = init_matrix(C1, C2, p, q,device, 'square_loss')
    
    tens=tensor_product(constC, hC1, hC2, T).to(device)
    
    return torch.sum(tens*T)

def _cost_matrix(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(-2)
    y_lin = y.unsqueeze(-3)
    C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
    return C
    
def entropic_gw(xs,xt,device,eps=1e-3,max_iter=100,verbose=False,log=True):

    C1 = _cost_matrix(xs,xs)
    C2 = _cost_matrix(xt,xt)
    p=torch.ones(xs.shape[0])/xs.shape[0]
    q=torch.ones(xt.shape[0])/xt.shape[0]
    p=p.to(device)
    q=q.to(device)
    if log:
        T,log=entropic_gromov_wasserstein(C1,C2,p,q,epsilon=eps,max_iter=max_iter,loss_fun='square_loss',verbose=verbose,device=device,log=True)
        return T,log
    else:
        T=entropic_gromov_wasserstein(C1,C2,p,q,epsilon=eps,max_iter=max_iter,loss_fun='square_loss',verbose=verbose,device=device,log=True)
        return T
    
    
def gwloss(constC, hC1, hC2, T):
    """ Return the Loss for Gromov-Wasserstein
    The loss is computed as described in Proposition 1 Eq. (6) in [12].
    Parameters
    ----------
    constC : ndarray, shape (ns, nt)
           Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
           h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
           h2(C) matrix in Eq. (6)
    T : ndarray, shape (ns, nt)
           Current value of transport matrix T
    Returns
    -------
    loss : float
           Gromov Wasserstein loss
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
    "Gromov-Wasserstein averaging of kernel and distance matrices."
    International Conference on Machine Learning (ICML). 2016.
    """

    tens = tensor_product(constC, hC1, hC2, T)

    return torch.sum(tens * T)

def init_matrix(C1, C2, p, q, device,loss_fun='square_loss'):
    """ Return loss matrices and tensors for Gromov-Wasserstein fast computation
    Returns the value of \mathcal{L}(C1,C2) \otimes T with the selected loss
    function as the loss function of Gromow-Wasserstein discrepancy.
    The matrices are computed as described in Proposition 1 in [12]
    Where :
        * C1 : Metric cost matrix in the source space
        * C2 : Metric cost matrix in the target space
        * T : A coupling between those two spaces
    The square-loss function L(a,b)=(1/2)*|a-b|^2 is read as :
        L(a,b) = f1(a)+f2(b)-h1(a)*h2(b) with :
            * f1(a)=(a^2)/2
            * f2(b)=(b^2)/2
            * h1(a)=a
            * h2(b)=b
    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
         Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
         Metric costfr matrix in the target space
    T :  ndarray, shape (ns, nt)
         Coupling between source and target spaces
    p : ndarray, shape (ns,)
    Returns
    -------
    constC : ndarray, shape (ns, nt)
           Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
           h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
           h2(C) matrix in Eq. (6)
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
    "Gromov-Wasserstein averaging of kernel and distance matrices."
    International Conference on Machine Learning (ICML). 2016.
    """

    if loss_fun == 'square_loss':
        def f1(a):
            return (a**2) / 2

        def f2(b):
            return (b**2) / 2

        def h1(a):
            return a

        def h2(b):
            return b
    elif loss_fun == 'kl_loss':
        raise NotImplementedError('Wait for it')

    constC1 = torch.matmul(torch.matmul(f1(C1), p.reshape(-1, 1)),
                     torch.ones(len(q)).to(device).reshape(1, -1))
    constC2 = torch.matmul(torch.ones(len(p)).to(device).reshape(-1, 1),
                     torch.matmul(q.reshape(1, -1), torch.transpose(f2(C2),1,0)))
    constC = constC1 + constC2
    hC1 = h1(C1)
    hC2 = h2(C2)

    return constC, hC1, hC2
    
    
def tensor_product(constC, hC1, hC2, T):
    """ Return the tensor for Gromov-Wasserstein fast computation
    The tensor is computed as described in Proposition 1 Eq. (6) in [12].
    Parameters
    ----------
    constC : ndarray, shape (ns, nt)
           Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
           h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
           h2(C) matrix in Eq. (6)
    Returns
    -------
    tens : ndarray, shape (ns, nt)
           \mathcal{L}(C1,C2) \otimes T tensor-matrix multiplication result
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
    "Gromov-Wasserstein averaging of kernel and distance matrices."
    International Conference on Machine Learning (ICML). 2016.
    """
    A = -torch.matmul(hC1, T).matmul(torch.transpose(hC2,1,0))
    tens = constC + A
    return tens
    
def gwggrad(constC, hC1, hC2, T):
    """ Return the gradient for Gromov-Wasserstein
    The gradient is computed as described in Proposition 2 in [12].
    Parameters
    ----------
    constC : ndarray, shape (ns, nt)
           Constant C matrix in Eq. (6)
    hC1 : ndarray, shape (ns, ns)
           h1(C1) matrix in Eq. (6)
    hC2 : ndarray, shape (nt, nt)
           h2(C) matrix in Eq. (6)
    T : ndarray, shape (ns, nt)
           Current value of transport matrix T
    Returns
    -------
    grad : ndarray, shape (ns, nt)
           Gromov Wasserstein gradient
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
    "Gromov-Wasserstein averaging of kernel and distance matrices."
    International Conference on Machine Learning (ICML). 2016.
    """
    return 2 * tensor_product(constC, hC1, hC2,
                              T)  # [12] Prop. 2 misses a 2 factor
    
    
def entropic_gromov_wasserstein(C1, C2, p, q, loss_fun, epsilon,device,
                            max_iter=100, tol=1e-6, verbose=False, log=False):
    """
    Returns the gromov-wasserstein transport between (C1,p) and (C2,q)
    (C1,p) and (C2,q)
    The function solves the following optimization problem:
    .. math::
        \GW = arg\min_T \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}-\epsilon(H(T))
        s.t. \GW 1 = p
             \GW^T 1= q
             \GW\geq 0
    Where :
        C1 : Metric cost matrix in the source space
        C2 : Metric cost matrix in the target space
        p  : distribution in the source space
        q  : distribution in the target space
        L  : loss function to account for the misfit between the similarity matrices
        H  : entropy
    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
         Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
         Metric costfr matrix in the target space
    p :  ndarray, shape (ns,)
         distribution in the source space
    q :  ndarray, shape (nt,)
         distribution in the target space
    loss_fun :  string
        loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float
        Regularization term >0
    max_iter : int, optional
       Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    Returns
    -------
    T : ndarray, shape (ns, nt)
        coupling between the two spaces that minimizes :
            \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}-\epsilon(H(T))
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
    "Gromov-Wasserstein averaging of kernel and distance matrices."
    International Conference on Machine Learning (ICML). 2016.
    """

    T = p[:,None]*q[None,:]
    
    constC, hC1, hC2 = init_matrix(C1, C2, p, q,device, loss_fun)
    
    cpt = 0
    err = 1
    
    if log:
        log = {'err': []}
    
    while (err > tol and cpt < max_iter):
    
        Tprev = T
    
        # compute the gradient
        tens = gwggrad(constC, hC1, hC2, T)
    
        T = sinkhorn(p, q, tens,device, epsilon)
    
        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            
            err = torch.norm(T - Tprev).item()
    
            if log:
                log['err'].append(err)
    
            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
    
        cpt += 1
        
    
    if log:
        log['loss'] = gwloss(constC, hC1, hC2, T)
        return T, log
    else:
        return T
        
def sinkhorn(p,q,C, device,epsilon=1e-3,threshold = 1e-1,numItermax=100):
            
    # Initialise approximation vectors in log domain
    u = torch.zeros_like(p).to(device)
    v = torch.zeros_like(q).to(device)

    # Stopping criterion
   
    # Sinkhorn iterations
    for i in range(numItermax): 
        u0, v0 = u, v
                    
        # u^{l+1} = a / (K v^l)
        K = _log_boltzmann_kernel(u, v, C,epsilon)
        u_ = torch.log(p + 1e-8) - torch.logsumexp(K, dim=1)
        u = epsilon * u_ + u
                    
        # v^{l+1} = b / (K^T u^(l+1))
        K_t = _log_boltzmann_kernel(u, v, C,epsilon).transpose(-2, -1)
        v_ = torch.log(q + 1e-8) - torch.logsumexp(K_t, dim=1)
        v = epsilon * v_ + v
        
        # Size of the change we have performed on u
        diff = torch.sum(torch.abs(u - u0), dim=-1) + torch.sum(torch.abs(v - v0), dim=-1)
        mean_diff = torch.mean(diff)
                    
        if mean_diff.item() < threshold:
            break
   
    # Transport plan pi = diag(a)*K*diag(b)
    K = _log_boltzmann_kernel(u, v, C,epsilon)
    pi = torch.exp(K)
    
    # Sinkhorn distance
    #cost = torch.sum(pi * C, dim=(-2, -1))

    return pi
    
def _log_boltzmann_kernel(u, v,C,epsilon):
    kernel = -C + u.unsqueeze(-1) + v.unsqueeze(-2)
    kernel /= epsilon
    return kernel
