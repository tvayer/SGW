#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:36:10 2019

@author: vayer
"""
import pylab as pl
import torch
import matplotlib as mpl
import argparse

#%%

if __name__ == '__main__':

    """ Plot the runtimes
    ----------
    path : path to the results previously calculated

    Example
    ----------
    python plot_runtimes.py -p '../res/runtime_2019_10_16_14_26_32/'
    
    """ 
    
    parser = argparse.ArgumentParser(description='Runtime')   
    parser.add_argument('-p','--path',type=str,help='Path to te results',required=True)
    args = parser.parse_args()    
    
    path=args.path
    
    d=torch.load(path+'runtime.pt',map_location='cpu')
    
    all_samples=d['all_samples']

    legen=[]
    
    fig = pl.figure(figsize=(15,8))
    ax = pl.axes()
        
    s=60
    fs=12
    colors={True:'r',False:'black'}
    
    ax.scatter(all_samples,d['t_all_gw'],c=[colors[x] for x in d['all_converged']],s=s)
    ax.plot(all_samples, d['t_all_gw'],'r')
    legen.append('Time GW POT')
    
    
    norm = mpl.colors.Normalize(vmin=0, vmax=3)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.YlOrBr)
    cmap.set_array([])
    
    norm = mpl.colors.Normalize(vmin=0, vmax=3)
    cmap2 = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Purples)
    cmap2.set_array([])
    
    for i,proj in enumerate(d['projs']):
        ax.scatter(all_samples,d[('t_all_sgw',proj)],s=s, c=cmap.to_rgba(i + 1))
        ax.plot(all_samples, d[('t_all_sgw',proj)], c=cmap.to_rgba(i + 1))
        legen.append('Time SGW Pytorch n_proj={}'.format(proj))
        
    for i,proj in enumerate(d['projs']):
        ax.scatter(all_samples,d[('t_all_sgw_numpy',proj)],s=s, c=cmap2.to_rgba(i + 1))
        ax.plot(all_samples, d[('t_all_sgw_numpy',proj)], c=cmap2.to_rgba(i + 1))
        legen.append('Time SGW numpy n_proj={}'.format(proj))
        
    ax.legend(legen,loc='upper left',fontsize=fs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    ax.set_ylabel('Seconds',fontsize=fs)
    ax.set_xlabel('Number of samples n in each distribution',fontsize=fs-3)
    
    ax.set_title('Running time',fontsize=fs)
    
    pl.xticks(fontsize=20)
    pl.yticks(fontsize=20)
        
    pl.title('Running time')
    
    pl.savefig('../res/running_times.pdf')
    pl.show()

