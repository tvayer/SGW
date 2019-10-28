#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 15:02:47 2019

@author: vayer
"""
import torch
import geoopt
import numpy as np
import torch.nn.functional as F
from sgw_pytorch import sgw_gpu
import time

def risgw_gpu(xs,xt,device,nproj=200,P=None,lr=0.001,
              max_iter=100, verbose=False, step_verbose=10, tolog=False, retain_graph=False):
    """ Returns RISGW between xs and xt eq (5) in [1]. 
    The dimension of xs must be less or equal than xt (ie p<=q)
    Parameters
    ----------
    xs : tensor, shape (n, p)
         Source samples
    xt : tensor, shape (n, q)
         Target samples (q>=p)
    device :  torch device
    nproj : integer
            Number of projections. Ignore if P is not None
    P : tensor, shape (max(p,q),n_proj)
        Projection matrix. If None creates a new projection matrix
    lr : float
            Learning rate for the optimization on Stiefel.
    max_iter : integer
            Maximum number of iterations for the gradient descent on Stiefel.            
    Returns
    -------
    C : tensor, shape (n_proj,1)
           Cost for each projection
    References
    ----------
    .. [1] Vayer Titouan, Chapel Laetitia, Flamary R{\'e}mi, Tavenard Romain
          and Courty Nicolas
          "Sliced Gromov-Wasserstein"
    Example
    ----------
    import numpy as np
    import torch
    from sgw_pytorch import sgw
    
    n_samples=300
    Xs=np.random.rand(n_samples,1)
    Xt=np.random.rand(n_samples,2)
    xs=torch.from_numpy(Xs).to(torch.float32)
    xt=torch.from_numpy(Xt).to(torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    P=np.random.randn(2,500)
    risgw_gpu(xs,xt,device,P=torch.from_numpy(P).to(torch.float32))    
    """ 
    affine_map = StiefelLinear(in_features=xs.size(1),
                               out_features=xt.size(1),device=device)
    
    optimizer = geoopt.optim.RiemannianAdam(affine_map.parameters(), lr=lr)
    
    log={}
    st=time.time()
    

    running_loss = 0.0
    for i in range(max_iter):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        loss = sgw_gpu(affine_map(xs), xt,device=device,nproj=nproj,tolog=False,P=P)
        loss.backward(retain_graph=retain_graph)
        optimizer.step()

        # print statistics
        running_loss = loss.item()
        if verbose and (i + 1) % step_verbose == 0:
            print('Iteration {}: sgw loss: {:.3f}'.format(i + 1,
                                                               running_loss))
    ed=time.time()
    if tolog:
        log['time']=ed-st
        log['Delta']=affine_map.weight.data
        return running_loss,log
    else:
        return running_loss
    

def stiefel_uniform_(tensor):  # TODO: make things better
    with torch.no_grad():
        tensor.data = torch.eye(tensor.size(0),tensor.size(1))
        return tensor

class StiefelLinear(torch.nn.Module):
    def __init__(self, in_features, out_features,device, bias=False):
        super(StiefelLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = geoopt.ManifoldParameter(
            data=torch.Tensor(out_features, in_features),
            manifold=geoopt.Stiefel()
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features).to(device))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
        self.weight.data=self.weight.data.to(device)

    def reset_parameters(self):
        stiefel_uniform_(self.weight)
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)
        

    
