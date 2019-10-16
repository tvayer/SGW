#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:53:30 2019

@author: vayer
"""
#%%

from sgw_pytorch import sgw_gpu
from sgw_numpy import sgw_cpu
from gw_pytorch import entropic_gw
import time
import ot
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from scipy.stats import special_ortho_group
import argparse
import torch
from ot.gromov import gromov_wasserstein as gw
import utils
#%%


def make_spiral(n_samples, noise=.5):
    n = np.sqrt(np.random.rand(n_samples,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_samples,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_samples,1) * noise
    return np.array(np.hstack((d1x,d1y)))
    
def create_ds(n_samples,spiral=False):
    if spiral:
        centers = [[5, 5], [-1, -1], [1, -1]]
        X, labels_true = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.2,
                                    random_state=0)
        Xs = X[labels_true==0,:]
        Xs = make_spiral(n_samples=n_samples, noise=1)
        
        centers = [[2, 2], [0, 0], [1, -1]]
        X, labels_true = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.2,
                                    random_state=0)
        Xt = X[labels_true==0,:]
        Xt = make_spiral(n_samples=n_samples, noise=1)
        
        A = special_ortho_group.rvs(2,random_state=0)
        
        get_rot= lambda theta : np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    
        A=get_rot(np.pi)
    
        Xt = np.dot(Xs,A)*.5+10*np.random.rand(1,2)+10
    else:
        Xs=(10-3)*np.random.rand(n_samples,2)+3
        Xt=(8-1)*np.random.rand(n_samples,2)+1
    
    return Xs,Xt
    
def calculate_sgw_numpy(Xs,Xt,nproj=200,tolog=False):
    if tolog:
        st=time.time()
        d_sgw,log_=sgw_cpu(Xs,Xt,nproj,tolog=tolog)
        ed=time.time()
        time_sgw=ed-st
    else:
        st=time.time()
        d_sgw=sgw_cpu(Xs,Xt,nproj,tolog=tolog)
        ed=time.time()
        time_sgw=ed-st
    if tolog :
        return d_sgw,time_sgw,log_
    else:
        return d_sgw,time_sgw
    
def calculate_sgw_gpu(Xs,Xt,device,nproj=200,tolog=False):
    xs=torch.from_numpy(Xs).to(torch.float32).to(device)
    xt=torch.from_numpy(Xt).to(torch.float32).to(device)
    if tolog:
        st=time.time()
        d_sgw,log_=sgw_gpu(xs,xt,device,nproj,tolog=tolog)
        ed=time.time()
        time_sgw=ed-st
    else:
        st=time.time()
        d_sgw=sgw_gpu(xs,xt,device,nproj,tolog=tolog)
        ed=time.time()
        time_sgw=ed-st
    if tolog :
        return d_sgw,time_sgw,log_
    else:
        return d_sgw,time_sgw
        
        
def calculate_entropic_gw_gpu(Xs,Xt,device,epsilon=1e-3,max_iter=100):
    xs=torch.from_numpy(Xs).to(torch.float32).to(device)
    xt=torch.from_numpy(Xt).to(torch.float32).to(device)
    
    st=time.time()
    d_entro_gw,log_=entropic_gw(xs,xt,device,eps=epsilon,max_iter=max_iter)
    ed=time.time()

    return d_entro_gw,ed-st,log_

        
def calculate_gw_cpu(Xs,Xt,numItermax=500,wass=False):
    st=time.time()
    C1=ot.dist(Xs,Xs,'sqeuclidean')
    C2=ot.dist(Xt,Xt,'sqeuclidean')
    p=np.ones(Xs.shape[0])/Xs.shape[0]
    q=np.ones(Xt.shape[0])/Xt.shape[0]
    T,log=gw(C1,C2,p,q,'square_loss',log=True,numItermax=numItermax)
    d_gw=log['loss'][::-1][0]
    converge_gw_pot=abs(log['loss'][::-1][0]-log['loss'][::-1][1])<=1e-5

    ed=time.time()
    time_gw=ed-st
    w_d=0
    t_w=0
    if wass:    
        st=time.time()
        M=ot.dist(Xs,Xt)
        w_d=ot.emd(p,q,M)
        ed=time.time()
        t_w=ed-st
    return d_gw,time_gw,converge_gw_pot,w_d,t_w
      
        


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Runtime')   
    parser.add_argument('-p','--log_dir',type=str,help='Path to write result',required=True)
    parser.add_argument('-ln','--all_samples', nargs='+',type=int, help='All L', required=True)
    parser.add_argument('-it','--nitermax',nargs='?',default=500,type=int,help='Iter Max GW POT')
    parser.add_argument('-ite','--it_entro',nargs='?',default=100,type=int,help='Iter Iter entropic gromov')

    parser.add_argument('-pr','--proj',nargs='+',default=50,type=int,help='Number of proj')
    parser.add_argument('-e','--eps',nargs='?',default=0.001,type=float,help='Epsilon for entropic gw')
   
    args = parser.parse_args()
    all_samples=args.all_samples
    numItermax=args.nitermax
    projs=args.proj
    epsilon=args.eps
    niterentro=args.it_entro
    onlyk=args.only_keras
    
    log_dir = utils.create_log_dir(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results={}
    

    results['all_samples']=all_samples
    results['projs']=projs
    results['epsilon']=epsilon

    results['all_gw']=[]
    results['t_all_gw']=[]
    results['all_w']=[]
    results['t_a_w']=[]
    results['all_converged']=[]

    results['all_entro_gw']=[]
    results['t_all_entro_gw']=[]

    for proj in projs:
        results[('t_all_sgw',proj)]=[]
        results[('all_sgw',proj)]=[]
        results[('all_log',proj)]=[]

        results[('t_all_sgw_numpy',proj)]=[]
        results[('all_sgw_numpy',proj)]=[]
        results[('all_log_numpy',proj)]=[]

        results[('t_all_sgw_k',proj)]=[]
        results[('all_sgw_k',proj)]=[]
        results[('all_log_k',proj)]=[]
                
    for n_samples in all_samples:
        Xs,Xt=create_ds(n_samples)
        
        d_gw,time_gw,converge_gw_pot,d_w,t_w=calculate_gw_cpu(Xs,Xt,numItermax,True)
        results['all_gw'].append(d_gw)
        results['t_all_gw'].append(time_gw)
        results['all_w'].append(d_w)
        results['t_a_w'].append(t_w)
        results['all_converged'].append(converge_gw_pot)
        
        if not onlyk:
            d_entro_gw,time_entro_gw,log_=calculate_entropic_gw_gpu(Xs,Xt,device,epsilon,niterentro)
            results['all_entro_gw'].append(d_entro_gw)
            results['t_all_entro_gw'].append(time_entro_gw)

        
        for proj in projs:
            
            d_sgw,time_sgw,log_=calculate_sgw_gpu(Xs,Xt,device,proj,True)
            results[('t_all_sgw',proj)].append(time_sgw)
            results[('all_sgw',proj)].append(d_sgw)
            results[('all_log',proj)].append(log_)
                
                
            d_sgw,time_sgw,log_=calculate_sgw_numpy(Xs,Xt,proj,True)
            results[('t_all_sgw_numpy',proj)].append(time_sgw)
            results[('all_sgw_numpy',proj)].append(d_sgw)
            results[('all_log_numpy',proj)].append(log_)

            print('n_samples={}, proj={} done... '.format(n_samples,proj))
            
            torch.cuda.empty_cache()
        
            torch.save(results,log_dir+'/runtime.pt')
