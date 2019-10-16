#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:45:05 2019

@author: rflamary
"""

import autograd.numpy as np
import autograd
import ot
import scipy as sp
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent


def get_P(d,L):
    res=np.random.randn(d,L)
    res/=np.sqrt(np.sum(res**2,0,keepdims=True))
    return res



def dist(x1, x2):
    """ Compute squared euclidean distance between samples (autograd)
    """
    x1p2 = np.sum(np.square(x1), 1)
    x2p2 = np.sum(np.square(x2), 1)
    return x1p2.reshape((-1, 1)) + x2p2.reshape((1, -1)) - 2 * np.dot(x1, x2.T)

def sw0(xs,xt,P): # xs.shape(1)<=xt.shape(1), P.shape[0]=xt.shape[1]
    
    L=P.shape[1]
    n=xs.shape[0]
    xsp=np.dot(xs,P[:xs.shape[1],:])
    xtp=np.dot(xt,P)
    
    #xsp=np.sort(xsp,0)
    #xtp=np.sort(xtp,0)
    
    res=0
    
    for l in range(L):
        
        x1=np.sort(xsp[:,l])
        x2=np.sort(xtp[:,l])
        
        D1=dist(x1[:,None],x2[:,None])
        
        l1=np.sum(np.square(D1))/n/n
        #l2=np.sum(D1[::-1,::-1])/n/n
        
        res+=l1
        
    return res/L

def sgw0(xs,xt,P): # xs.shape(1)<=xt.shape(1), P.shape[0]=xt.shape[1]
    
    L=P.shape[1]
    n=xs.shape[0]
    xsp=np.dot(xs,P[:xs.shape[1],:])
    xtp=np.dot(xt,P)
    
    #xsp=np.sort(xsp,0)
    #xtp=np.sort(xtp,0)
    
    res=0
    
    for l in range(L):
        
        x1=np.sort(xsp[:,l])
        x2=np.sort(xtp[:,l])
        
        D1=dist(x1[:,None],x1[:,None])
        D2=dist(x2[:,None],x2[:,None])
        
        l1=np.sum(np.square(D1-D2))/n/n
        l2=np.sum(np.square(D1[::-1,::-1]-D2))/n/n
        
        res+=np.minimum(l1,l2)
        
    return res/L

def sgw_bary(Xs,ws,P,X0,weights=None,nitermax=10):
    
    shp=X0.shape
    
    def loss(x):
        res=0
        for xi,wi in zip(Xs,ws):
            res+=sgw0(x.reshape(shp),xi,P) 
        return res
    
    #print(loss(X0))
    
    grad=autograd.grad(loss)
    
    sol=sp.optimize.minimize(loss,X0,jac=grad,method='L-BFGS-B')
    
    return sol['x'].reshape(shp)
    
def risgw_bary(Xs,ws,P,X0,weights=None,nitermax=10):
    
    shp=X0.shape
    
    def update_x(X0,deltas):
    
        def loss(x):
            res=0
            for xi,deltai in zip(Xs,deltas):
                res+=sgw0(np.dot(np.reshape(x,shp),deltai),xi,P) 
            return res
        
        print(loss(X0))
        
        grad=autograd.grad(loss)
        
        sol=sp.optimize.minimize(loss,X0,jac=grad,method='L-BFGS-B')
        
        return sol['fun'],sol['x'].reshape(shp)
    
    def update_deltas(X0,deltas):
        
        deltasr=[]
        loss=0
        for xi,deltai in zip(Xs,deltas):
            temp,delta=risgw2(X0,xi,P,X0=deltai)
            deltasr.append(delta)
            loss=loss+temp
        return loss,deltasr
    
    deltas=[1.0*np.eye(X0.shape[1]) for x in Xs]
    
    
    for loop in range(nitermax):
        
        lossd,deltas=update_deltas(X0,deltas)
        
        print('Update D: ',lossd)
        
        lossx,X0=update_x(X0,deltas)
        
        print('Update X: ',lossx)            
                     
        
    
    return X0
        


def risgw(xs,xt,P):
    
    def loss(delta):
        return sgw0(np.dot(xs,delta),xt,P)
    
    manifold = Stiefel(xt.shape[1], xs.shape[1])

    
    problem = Problem(manifold=manifold, cost=loss)
    
    # (3) Instantiate a Pymanopt solver
    solver = SteepestDescent(logverbosity=0)
    
    # let Pymanopt do the rest
    Xopt = solver.solve(problem,x=np.eye( xs.shape[1],xt.shape[1]))
    
    return loss(Xopt)
        
def risgw2(xs,xt,P,X0=None):
    
    def loss(delta):
        return sgw0(np.dot(xs,delta),xt,P)
    
    if X0 is None:
        X0=np.eye( xs.shape[1],xt.shape[1])
        
    
    manifold = Stiefel(xt.shape[1], xs.shape[1])

    
    problem = Problem(manifold=manifold, cost=loss)
    
    # (3) Instantiate a Pymanopt solver
    solver = SteepestDescent(logverbosity=0)
    
    # let Pymanopt do the rest
    Xopt = solver.solve(problem,x=X0)
    
    return loss(Xopt), Xopt

def risw(xs,xt,P):
    
    def loss(delta):
        return sw0(np.dot(xs,delta),xt,P)
    
    manifold = Stiefel(xt.shape[1], xs.shape[1])

    
    problem = Problem(manifold=manifold, cost=loss)
    
    # (3) Instantiate a Pymanopt solver
    solver = SteepestDescent(logverbosity=0)
    
    # let Pymanopt do the rest
    Xopt = solver.solve(problem,x=np.eye( xs.shape[1],xt.shape[1]))
    
    return loss(Xopt)
        
def risw2(xs,xt,P,X0=None):
    
    def loss(delta):
        return sw0(np.dot(xs,delta),xt,P)
    
    if X0 is None:
        X0=np.eye( xs.shape[1],xt.shape[1])
        
    
    manifold = Stiefel(xt.shape[1], xs.shape[1])

    
    problem = Problem(manifold=manifold, cost=loss)
    
    # (3) Instantiate a Pymanopt solver
    solver = SteepestDescent(logverbosity=0)
    
    # let Pymanopt do the rest
    Xopt = solver.solve(problem,x=X0)
    
    return loss(Xopt), Xopt
    
    
def gw0(xs,xt): # xs.shape(1)<=xt.shape(1), P.shape[0]=xt.shape[1]
    
    n=xs.shape[0]
    u=ot.unif(n)
    D1=dist(xs,xs)
    D2=dist(xt,xt)
        
    return ot.gromov.gromov_wasserstein2(D1,D2,u,u,'square_loss')[0]

def w0(xs,xt): # xs.shape(1)<=xt.shape(1), P.shape[0]=xt.shape[1]
    
    n=xs.shape[0]
    u=ot.unif(n)
    D1=dist(xs,xt)
        
    return ot.emd2(u,u,D1)
    
