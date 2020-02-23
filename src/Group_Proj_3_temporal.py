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


# No. of elements
N_el = 10
# Length of Rod
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
Mg = np.zeros((N_el+1,N_el+1))
Mg_upper_diag = h/6*np.ones(N_el)
Mg_diag = 2*h/3*np.ones(N_el+4)
Mg_diagonals = [Mg_upper_diag, Mg_diag, Mg_upper_diag]
Mg = diags(Mg_diagonals, [-1,0,1])
Mg = sp.sparse.csr_matrix(Mg)
# Replace first/last row
Mg[0,0] = 1
Mg[0,1] = 0
Mg[-1,-1] = 1
Mg[-1,-2] = 0

# Building Le_block
Lg = np.zeros((N_el+1,N_el+1))
Lg_upper_diag = -1/h*np.ones(N_el)
Lg_diag = 2/h*np.ones(N_el+4)
Lg_diagonals = [Lg_upper_diag, Lg_diag, Lg_upper_diag]
Lg = diags(Lg_diagonals, [-1,0,1])
Lg = sp.sparse.csr_matrix(Lg)
# Replace first/last row
Lg[0,0] = 1
Lg[0,1] = 0
Lg[-1,-1] = 1
Lg[-1,-2] = 0

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

# Imposing Initial Condition for Consistent Matrix
u_old = 200*np.ones(N_nodes)
uD = np.zeros(N_nodes)
RHS = np.zeros(N_nodes)
CFL =  0.02
dt = CFL*h
t_steps = 1000
k = 0

# Lumped Solutions
Mg_lumped = np.zeros((N_nodes,N_nodes))
u_lumped = np.zeros(N_nodes)
u_old_lumped = 200*np.ones(N_nodes)
RHS_lumped = np.zeros(N_nodes)
for i in range(N_nodes):
    Mg_lumped[i,i] = h
Mg_lumped[0,0] = 1
Mg_lumped[-1,-1] = 1
    
# Analytical Solution
a = 100*(math.exp(L_right)-math.exp(L_left))/L;
b = 200 + 100*math.exp(L_right) -a*L_right;
T_analy = np.zeros(N_nodes)
for i in range(N_nodes):
    T_analy[i] = -100*math.exp(x[i]) + a*x[i] + b
    

while k <= t_steps:
    # Forming RHS of inverse problem
    RHS = Mg @ u_old + ( dt * F_g) - dt * Lg @ u_old
    RHS_lumped = Mg_lumped @ u_old_lumped + ( dt * F_g) - dt * Lg @ u_old_lumped
    # Solving system
    uD = inv(Mg) @ RHS
    u_lumped = np.linalg.inv(Mg_lumped) @ RHS_lumped
    # Assigning New Temperature Distribution to 'Old' Variables for next t-step
    u_old = uD
    u_old_lumped = u_lumped
    if ((k%2) == 0):
        plt.plot(x,uD, '-o', label='Consistent')
        plt.plot(x,u_lumped, '-o', label = 'Lumped')
        plt.plot(x,T_analy, '--', label = 'Analytical')
        plt.title('Timestamp: ' + str(round(k*dt,4)) +'s' + '   Total Time: ' + str(round(t_steps*dt,4))   )
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('T')
        plt.grid()
        plt.pause(0.001)
        if (k != t_steps):
            plt.clf()
        
    k += 1

plt.figure()
plt.plot(x,uD - T_analy, '-o', label="Error of Consistent Method")
plt.plot(x,u_lumped - T_analy, '-o', label="Error of Lumped Method")
plt.legend()
plt.xlabel('x')
plt.ylabel('error')
plt.grid()

    