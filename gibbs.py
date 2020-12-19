# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 16:12:20 2020

@author: jvsriram
"""
import numpy as np

W = np.array([0,1,2,3,4])  #vocab

Docs = np.array([
    [3,1,4,3,2,0,2],
    [4,1,2,4,3,2,0],                #D->documents which is represented as rows
    [4,2,3,4,1,2,0],                #each row is a different document
    [1,1,0,0,3,4,4],
])

N_D = Docs.shape[0]  # num of docs
N_W = W.shape[0]  # num of words
N_K = 3  # num of topics 
#topics start from 0 to N_K - 1

print('Documents:')
print(Docs)
print("\n")
alpha = 1
beta = 1

Z = np.zeros(shape=[N_D, N_W])      # Z := word topic assignment
for i in range(N_D):
    for l in range(N_W):
        Z[i, l] = np.random.randint(N_K)  # randomly assign word's topic

print("initial random assignment of toPhics:\n",Z,"\n")
Phi = np.zeros([N_D, N_K])                     # document topic distribution
for i in range(N_D):
    Phi[i] = np.random.dirichlet(alpha*np.ones(N_K))


B = np.zeros([N_K, N_W])                          # word topic distribution
for k in range(N_K):
    B[k] = np.random.dirichlet(beta*np.ones(N_W))


# Gibbs sampling
for e in range(1000):
    for i in range(N_D):
        for l in range(N_W):
            p_bar_il = np.exp(np.log(Phi[i]) + np.log(B[:, Docs[i, l]]))
            p_il = p_bar_il / np.sum(p_bar_il)            
            z_il = np.random.multinomial(1, p_il)           # Resample word topic assignment Z
            Z[i, l] = np.argmax(z_il)

    for i in range(N_D):
        m = np.zeros(N_K)
        for k in range(N_K):
            m[k] = np.sum(Z[i] == k)

        # Resample doc topic matrix
        Phi[i, :] = np.random.dirichlet(alpha + m)

    for k in range(N_K):
        n = np.zeros(N_W)
        for v in range(N_W):
            for i in range(N_D):
                for l in range(N_W):
                    n[v] += (Docs[i, l] == v) and (Z[i, l] == k)

        # Resample word topic matrix
        B[k, :] = np.random.dirichlet(beta + n)


print('Document topic distribution:')
print(Phi)
print("\n")
print('Topic\'s word distribution:')
print(B)
print("\n")
print('Word topic assignment:')
print(Z)
