#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 16:40:24 2019

@author: rflamary
"""

import sys
sys.path.append('../lib')

import numpy as np
import pylab as pl
import sgw_numpy2 as sg

#%%
def make_spiral(n_samples, noise=.5):
    n = np.sqrt(np.random.rand(n_samples,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_samples,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_samples,1) * noise
    return np.array(np.hstack((d1x,d1y)))

get_rot= lambda theta : np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])

def get_data(n_samples,theta,scale=1,transla=0):
    Xs = make_spiral(n_samples=n_samples, noise=1)-transla
    Xt = make_spiral(n_samples=n_samples, noise=1)
    
    A=get_rot(theta)
    
    Xt = (np.dot(Xt,A))*scale+transla
    
    return Xs,Xt
   

n_samples=300
theta=0#p.pi/2
scale=1

Xs,Xt=get_data(n_samples,theta,scale=scale)

temp,Xt2=get_data(n_samples,np.pi/2,scale=scale)
temp,Xt22=get_data(n_samples,np.pi/4,scale=scale)

temp,Xt3=get_data(n_samples,0,scale=1.5)
Xt2[:,0]+=70
Xt22[:,0]+=35
Xt3[:,1]+=40
#%%

a=0.5
pl.figure(2,figsize=(6,3))
pl.clf()
pl.scatter(Xs[:,0],Xs[:,1],marker='o',s=40,edgecolors='k',alpha=a,label='Source')
pl.scatter(Xt[:,0],Xt[:,1],marker='o',s=40,edgecolors='k',alpha=a,label="Target (rot=0)")
pl.scatter(Xt22[:,0],Xt22[:,1],marker='o',s=40,edgecolors='k',alpha=a,label="Target (rot=$\pi/4$)")
pl.scatter(Xt2[:,0],Xt2[:,1],marker='o',s=40,edgecolors='k',alpha=a,label="Target (rot=$\pi/2$)")
pl.xticks([])
pl.yticks([])
pl.title('Spiral datasets used for experiments')
pl.legend()
pl.show()
pl.savefig('../res/spiral_exemple.pdf',bbox_inches='tight')

#%%
fname="../res/spirale_rot2_L{}_nbloop{}_nbrot{}.npz"
nbrot=10
nbloop=5
angles=np.linspace(0,np.pi/2,nbrot)
scale=1
L=20

#%%

GW=np.zeros((nbloop,nbrot))
SW=np.zeros((nbloop,nbrot))
W=np.zeros((nbloop,nbrot))

SGW=np.zeros((nbloop,nbrot))
RISGW=np.zeros((nbloop,nbrot))
RISW=np.zeros((nbloop,nbrot))

rota={}


for i in range(nbloop):
    
    P=sg.get_P(2,L)
    Xs,Xt0=get_data(n_samples,0,scale=scale)
    
    for j,theta in enumerate(angles):
        
        A=get_rot(theta)
        Xt=Xt0.dot(A)
                
        GW[i,j]=sg.gw0(Xs,Xt)
        
        SGW[i,j]=sg.sgw0(Xs,Xt,P)
                
        RISGW[i,j]=sg.risgw(Xs,Xt,P)
                
        print('--------------{0:.2f} Done--------------'.format(100*j/len(angles)))
        
        np.savez(fname.format(L,nbloop,nbrot),angles=angles,GW=GW,SGW=SGW,RISGW=RISGW,
                 SW=SW,W=W,RISW=RISW)
    print('!!!!!!!!!!!!!!!!{0} Loop Done!!!!!!!!!!!!!!!!'.format(i))
    
#%%   
a=0.5
pl.figure(2,figsize=(6,3))
pl.clf()
pl.scatter(Xs[:,0],Xs[:,1],marker='o',s=40,edgecolors='k',alpha=a,label='Source',c='red')
pl.scatter(Xt0[:,0],Xt0[:,1],marker='o',s=40,edgecolors='k',alpha=a,label="Target (rot=0)",c='blue')
Xt_rotat=np.dot(Xt0,rota[(i,0)])
pl.scatter(Xt_rotat[:,0],Xt_rotat[:,1],marker='o',s=40,edgecolors='k',alpha=a,label="Projected target (rot=0)",c='orange')
pl.xticks([])
pl.yticks([])
pl.title('Spiral datasets used for experiments')
pl.legend()
pl.show()
    
#%%
res=np.load(fname.format(L,nbloop,nbrot))     
angles=res["angles"]

GW=res["GW"]
SGW=res["SGW"]
RISGW=res["RISGW"]

    

def plot_perf(nlist,err,color,label,errbar=False,perc=20):
    pl.plot(nlist,err.mean(0),label=label,color=color)
    if errbar:
        pl.fill_between(nlist,np.percentile(err,perc,axis=0),np.percentile(err,100-perc,axis=0),
                    alpha=0.2,facecolor=color)

do_err=True
pl.figure(1,(4,4))

pl.clf()
plot_perf(angles,GW,'r','GW',do_err)     
plot_perf(angles,SGW,'g','SGW',do_err)    
plot_perf(angles,RISGW,'k','RISGW',do_err) 

 
pl.title("Values for increasing rotation")
pl.grid()     
pl.xlabel('Rotation angle (radian)')

pl.xticks((0,np.pi/8,np.pi/4,3*np.pi/8,np.pi/2),('0','$\pi/8$','$\pi/4$','$3\pi/8$','$\pi/2$'))
pl.ylabel('Value')
pl.legend()
pl.savefig('../res/spiral_rot.pdf',bbox_inches='tight')


