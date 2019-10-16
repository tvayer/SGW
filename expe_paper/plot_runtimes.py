#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:36:10 2019

@author: vayer
"""
import pylab as pl
import torch
import matplotlib as mpl

#%%
path='../res/runtime_2019_04_23_14_56_55/'

d=torch.load(path+'runtime.pt',map_location='cpu')

all_samples=d['all_samples']

#%%
legen=[]

pl.figure(figsize=(15,8))

pl.subplot(1,2,1)

s=60
fs=12
colors={True:'r',False:'black'}

pl.scatter(all_samples,d['t_all_gw'],c=[colors[x] for x in d['all_converged']],s=s)
pl.plot(all_samples, d['t_all_gw'],'r')
legen.append('Time GW POT')


norm = mpl.colors.Normalize(vmin=0, vmax=3)
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.YlOrBr)
cmap.set_array([])

norm = mpl.colors.Normalize(vmin=0, vmax=3)
cmap2 = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Purples)
cmap2.set_array([])

for i,proj in enumerate(d['projs']):
    pl.scatter(all_samples,d[('t_all_sgw',proj)],s=s, c=cmap.to_rgba(i + 1))
    pl.plot(all_samples, d[('t_all_sgw',proj)], c=cmap.to_rgba(i + 1))
    legen.append('Time SGW n_proj={}'.format(proj))
    
for i,proj in enumerate(d['projs']):
    pl.scatter(all_samples,d[('t_all_sgw_numpy',proj)],s=s, c=cmap2.to_rgba(i + 1))
    pl.plot(all_samples, d[('t_all_sgw_numpy',proj)], c=cmap2.to_rgba(i + 1))
    legen.append('Time SGW numpy n_proj={}'.format(proj))
    
pl.legend(legen,loc='upper left',fontsize=fs)
pl.ylabel('Time in s. Semi log axis')
pl.xlabel('Number of samples in each distri')
pl.xlim(0,max(all_samples)+10)

pl.subplot(1,2,2)
s=60
colors={True:'r',False:'black'}

pl.scatter(all_samples,d['t_all_gw'],c=[colors[x] for x in d['all_converged']],s=s)
pl.plot(all_samples, d['t_all_gw'],'r')


for i,proj in enumerate(d['projs']):
    pl.scatter(all_samples,d[('t_all_sgw',proj)],s=s, c=cmap.to_rgba(i + 1))
    pl.plot(all_samples, d[('t_all_sgw',proj)], c=cmap.to_rgba(i + 1))

for i,proj in enumerate(d['projs']):
    pl.scatter(all_samples,d[('t_all_sgw_numpy',proj)],s=s, c=cmap2.to_rgba(i + 1))
    pl.plot(all_samples, d[('t_all_sgw_numpy',proj)], c=cmap2.to_rgba(i + 1))
pl.ylabel('Time in s')
pl.xlabel('Number of samples in each distri')
pl.xlim(0,max(all_samples)+10)
pl.ylim(0,0.1)
#pl.yscale('symlog')

pl.suptitle('Running time')

pl.savefig('../res/running_times.pdf')
pl.show()

