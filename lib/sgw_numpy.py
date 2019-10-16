#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:04:34 2019

@author: vayer
"""

import numpy as np
import time 

class BadShapeError(Exception):
    pass 


def sgw_cpu(xs,xt,nproj=200,tolog=False,P=None):
    """ Returns SGW between xs and xt eq (4) in [1]. Only implemented with the 0 padding Delta.
    Parameters
    ----------
    xs : numpy array, shape (n, p)
         Source samples
    xt : numpy array, shape (n, q)
         Target samples
    nproj : integer
            Number of projections. Ignore if P is not None
    P : numpy array, shape (max(p,q),n_proj)
        Projection matrix. If None creates a new projection matrix
    tolog : bool
            Wether to return timings or not
    Returns
    -------
    C : numpy array, shape (n_proj,1)
           Cost for each projection
    References
    ----------
    .. [1] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
          "Sliced Gromov-Wasserstein"
    Example
    ----------
    import numpy as np
    from sgw_numpy import sgw_cpu
    
    n_samples=300
    Xs=np.random.rand(n_samples,2)
    Xt=np.random.rand(n_samples,1)
    P=np.random.randn(2,500)
    sgw_cpu(Xs,Xt,P=P)
    """   
    
    if tolog:
        log={}

    if tolog: 
        st=time.time()
        xsp,xtp=sink_(xs,xt,nproj,P)
        ed=time.time()   
        log['time_sink_']=ed-st
    else:
        xsp,xtp=sink_(xs,xt,nproj,P)
    if tolog:    
        st=time.time()
        d,log_gw1d=gromov_1d(xsp,xtp,tolog=True)
        ed=time.time()   
        log['time_gw_1D']=ed-st
        log['gw_1d_details']=log_gw1d
    else:
        d=gromov_1d(xsp,xtp,tolog=False)
    
    if tolog:
        return d,log
    else:
        return d

def _cost(xsp,xtp,tolog=False):
    """ Returns the GM cost eq (3) in [1]
    Parameters
    ----------
    xsp : tensor, shape (n, n_proj)
         1D sorted samples (after finding sigma opt) for each proj in the source
    xtp : tensor, shape (n, n_proj)
         1D sorted samples (after finding sigma opt) for each proj in the target
    tolog : bool
            Wether to return timings or not
    Returns
    -------
    C : tensor, shape (n_proj,1)
           Cost for each projection
    References
    ----------
    .. [1] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
          "Sliced Gromov-Wasserstein"
    """
    st=time.time()
    allC=[]
    for j in range(xsp.shape[1]):
        xs=xsp[:,j]
        xt=xtp[:,j]
        
        X=np.sum(xs)
        X2=np.sum(xs**2)
        X3=np.sum(xs**3)
        X4=np.sum(xs**4)
        
        Y=np.sum(xt)
        Y2=np.sum(xt**2)
        Y3=np.sum(xt**3)
        Y4=np.sum(xt**4)
        
        xxyy_=np.sum((xs**2)*(xt**2))
        xxy_=np.sum((xs**2)*(xt))
        xyy_=np.sum((xs)*(xt**2))
        xy_=np.sum((xs)*(xt))
        
                
        n=xs.shape[0]

        C2=2*X2*Y2+2*(n*xxyy_-2*Y*xxy_-2*X*xyy_+2*xy_**2)

        power4_x=2*n*X4-8*X3*X+6*X2**2
        power4_y=2*n*Y4-8*Y3*Y+6*Y2**2

        C=(1/(n**2))*(power4_x+power4_y-2*C2)
        
        allC.append(C)
        
    ed=time.time()
    
    if not tolog:
        return allC 
    else:
        return allC,ed-st

    

def gromov_1d(xs,xt,tolog=False,fast=True):
    """ Solves the Gromov in 1D (eq (2) in [1] for each proj
    Parameters
    ----------
    xsp : tensor, shape (n, n_proj)
         1D sorted samples for each proj in the source
    xtp : tensor, shape (n, n_proj)
         1D sorted samples for each proj in the target
    tolog : bool
            Wether to return timings or not
    fast: use the O(nlog(n)) cost or not
    Returns
    -------
    toreturn : tensor, shape (n_proj,1)
           The SGW cost for each proj
    References
    ----------
    .. [1] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
          "Sliced Gromov-Wasserstein"
    """  
    if tolog:
        log={}
    
    st=time.time()
    xs2=np.sort(xs,axis=0)
    
    if tolog:
        xt_asc=np.sort(xt,axis=0)
        xt_desc=np.sort(xt,axis=0)[::-1]
        l1,t1=_cost(xs2,xt_asc,tolog=tolog)
        l2,t2=_cost(xs2,xt_desc,tolog=tolog)
    else:
        xt_asc=np.sort(xt,axis=0)
        xt_desc=np.sort(xt,axis=0)[::-1]
        l1=_cost(xs2,xt_asc,tolog=tolog)
        l2=_cost(xs2,xt_desc,tolog=tolog)            
    toreturn=np.mean(np.minimum(l1,l2))
    
    ed=time.time()
            
   
    if tolog:
        log['g1d']=ed-st
        log['t1']=t1
        log['t2']=t2
 
    if tolog:
        return toreturn,log
    else:
        return toreturn
        

def sink_(xs,xt,nproj=200,P=None):
    """ Sinks the points of the measure in the lowest dimension onto the highest dimension and applies the projections.
    Only implemented with the 0 padding Delta=Delta_pad operator (see [1])
    Parameters
    ----------
    xs : tensor, shape (n, p)
         Source samples
    xt : tensor, shape (n, q)
         Target samples
    device :  torch device
    nproj : integer
            Number of projections. Ignored if P is not None
    P : tensor, shape (max(p,q),n_proj)
        Projection matrix
    Returns
    -------
    xsp : tensor, shape (n,n_proj)
           Projected source samples 
    xtp : tensor, shape (n,n_proj)
           Projected target samples 
    References
    ----------
    .. [1] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
          "Sliced Gromov-Wasserstein"
    """  
    dim_d= xs.shape[1]
    dim_p= xt.shape[1]
    
    if dim_d<dim_p:
        random_projection_dim = dim_p
        topad = np.zeros((xs.shape[0],dim_p-dim_d))
        xs2=np.concatenate((xs,topad),axis=1)
        xt2=xt
    else:
        random_projection_dim = dim_d
        topad = np.zeros((xt.shape[0],dim_d-dim_p))
        xt2=np.concatenate((xt,topad),axis=1)
        xs2=xs
        
    if P is None:
        P=np.random.randn(random_projection_dim,nproj)
    p=P/np.sqrt(np.sum(P**2,axis=0,keepdims=True))

        
    try:   
        xsp=np.dot(xs2,p)
        xtp=np.dot(xt2,p)
    except ValueError as error:
        print('----------------------------------------')
        print('xs origi dim :', xs.shape)
        print('xt origi dim :', xt.shape)
        print('dim_p :', dim_p)
        print('dim_d :', dim_d)
        print('random_projection_dim : ',random_projection_dim)
        print('projector dimension : ',p.shape)
        print('xs2 dim :', xs2.shape)
        print('xt2 dim :', xt2.shape)
        print('xs_tmp dim :', xs2.shape)
        print('xt_tmp dim :', xt2.shape)
        print('----------------------------------------')
        print(error)
        raise BadShapeError
    
    return xsp,xtp