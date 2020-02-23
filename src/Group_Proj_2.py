#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 22:01:28 2020

@author: chihin
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
from scipy.sparse import diags
from scipy.linalg import block_diag

N_el = 50
N_nodes = N_el +1
L = 4
L_left = -L/2
L_right = L/2
x = np.linspace(-L/2,L/2,N_nodes)
h = L/N_el
P = 1
N_eof = N_el*(P+1)
N_dof = N_el+1

N_loc = P+1 # Nodes per element, 2 since linear 


# Building Me_block
Me_block = np.zeros((N_el+1,N_el+1))
Me_upper_diag = h/6*np.ones(N_el)
Me_diag = 2*h/3*np.ones(N_el+4)
Me_diagonals = [Me_upper_diag, Me_diag, Me_upper_diag]
Me_block = diags(Me_diagonals, [-1,0,1])
Me_block = sp.sparse.csr_matrix(Me_block)
# Replace first/last row
Me_block[0,0] = 1
Me_block[0,1] = 0
Me_block[-1,-1] = 1
Me_block[-1,-2] = 0

# Building Le_block
Le_block = np.zeros((N_el+1,N_el+1))
Le_upper_diag = -1/h*np.ones(N_el)
Le_diag = 2/h*np.ones(N_el+4)
Le_diagonals = [Le_upper_diag, Le_diag, Le_upper_diag]
Le_block = diags(Le_diagonals, [-1,0,1])
Le_block = sp.sparse.csr_matrix(Le_block)
# Replace first/last row
Le_block[0,0] = 1
Le_block[0,1] = 0
Le_block[-1,-1] = 1
Le_block[-1,-2] = 0

# Building Assembly Matrix
A_inner = np.kron(np.eye(N_el-1), np.ones((2,1)))
A = block_diag(1,A_inner,1)


# Constructing Source
F_e = np.zeros(N_eof)
f = np.zeros(N_nodes)

source = lambda x : 100*math.exp(x)
for i in range(N_nodes):
    f[i] = source(i*h)

for i in range(N_el):
    F_e[2*i] = 100/h * ( math.exp(x[i+1]) - (h+1)*math.exp(x[i]) )
    F_e[2*i+1] = 100/h * ( (h-1)*math.exp(x[i+1]) + math.exp(x[i]) )    

F_g = np.transpose(A)@F_e
F_g[0] = 200
F_g[-1] = 200

# Imposing Dirichlet Condition

uD = np.zeros(N_nodes)
uD = np.linalg.inv(Le_block.todense())@F_g

plt.plot(x,np.transpose(uD), '-o')
plt.grid()
plt.xlabel('x')
plt.ylabel('T')
