#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 00:34:01 2020

@author: chihin
"""

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
from scipy.sparse.linalg import inv
from scipy.linalg import block_diag

N_el = 5
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
Mg_block = np.zeros((N_el+1,N_el+1))
Mg_upper_diag = h/6*np.ones(N_el)
Mg_diag = 2*h/3*np.ones(N_el+4)
Mg_diagonals = [Me_upper_diag, Me_diag, Me_upper_diag]
Mg_block = diags(Me_diagonals, [-1,0,1])
Mg_block = sp.sparse.csr_matrix(Me_block)
# Replace first/last row
Mg_block[0,0] = 1
Mg_block[0,1] = 0
Mg_block[-1,-1] = 1
Mg_block[-1,-2] = 0

# Building Le_block
Lg_block = np.zeros((N_el+1,N_el+1))
Lg_upper_diag = -1/h*np.ones(N_el)
Lg_diag = 2/h*np.ones(N_el+4)
Lg_diagonals = [Le_upper_diag, Le_diag, Le_upper_diag]
Lg_block = diags(Le_diagonals, [-1,0,1])
Lg_block = sp.sparse.csr_matrix(Le_block)
# Replace first/last row
Lg_block[0,0] = 1
Lg_block[0,1] = 0
Lg_block[-1,-1] = 1
Lg_block[-1,-2] = 0

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

# Imposing Initial Condition
u_old = 1*np.ones(N_nodes)
uD = np.zeros(N_nodes)
RHS = np.zeros(N_nodes)
dt = 0.01
t_steps = 5
k = 0

while k <= t_steps:
    RHS = Mg_block @ u_old + ( dt * F_g) + dt * Lg_block @ u_old
    k += 1
    uD = inv(Mg_block) @ RHS
    u_old = uD
    plt.plot(x,uD)
    