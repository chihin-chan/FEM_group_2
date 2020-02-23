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

# Specifying Number of Elements
N_el = 5 
# Specify Length of Rod
L = 4

N_nodes = N_el +1
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

# Analytical Solution
a = 100*(math.exp(L_right)-math.exp(L_left))/L;
b = 200 + 100*math.exp(L_right) -a*L_right;
T_analy = np.zeros(N_nodes)
for i in range(N_nodes):
    T_analy[i] = -100*math.exp(x[i]) + a*x[i] + b
    

plt.plot(x,np.transpose(uD), '-o', label='FEM')
plt.plot(x, T_analy, '--', label='Analytical')
plt.title('Temperature Distribution \n A comparison between FEM and analytical solution.')
plt.legend()
plt.grid()
plt.xlabel('x')
plt.ylabel('T')

error = np.zeros(N_nodes)

for i in range(N_nodes):
    error [i] = uD[0,i] - T_analy[i]

plt.figure()
plt.plot(x, error, '-o', label="Error of FEM")
plt.title("Error distribution along rod. \n It is so small which is \n probably due to round-off errors")
plt.legend()
plt.xlabel('x')
plt.ylabel('error')
plt.grid()
