# Load libraries
import numpy as np
from LDA_AandB.Optimization_Examples_JIT import Loop3, Loop4

def initialize(V = 50, N = 300, K = 3, M = 2):
    """ Create toy data for timing examples"""
    
    # Set true parameters
    alpha_true = np.random.randint(1, 15, K)
    beta_true = np.random.randint(1, 10, V)
    
    # Generate data
    phi_true = np.zeros((K, V))
    for k in range(K):
        phi_true[k, :] = np.random.dirichlet(beta_true)
     
    theta_true = np.zeros((M, K))
    for m in range(M):
        theta_true[m,:] = np.random.dirichlet(alpha_true)
        
    doc_lens = np.random.randint(100, N, M)
    z_true = {}
    w = {}
    for m in range(M):
        z_true[m] = []
        w[m] = []
        for n in range(doc_lens[m]):
            z_true[m].extend(np.nonzero(np.random.multinomial(1, theta_true[m,:]))[0])
            w[m].extend(np.nonzero(np.random.multinomial(1, phi_true[z_true[m][n], :]))[0])
            
    # Set initial z randomly
    z = {}
    for m in range(M):
        z[m] = []
        for n in range(doc_lens[m]):
            z[m].extend(np.nonzero(np.random.multinomial(1, np.ones(K)/K))[0])
            
            
    # Create count matrices
    A = np.zeros((M, K))
    for m in range(M):
        for k in range(K):
            A[m, k] = sum(np.array(z[m]) == k)
            
    B = np.zeros((K, V))
    for m in range(M):
        for n in range(doc_lens[m]):
            B[z[m][n], w[m][n]] += 1         
            
    C = np.zeros(K)
    for m in range(M):
        for n in range(doc_lens[m]):
            C[z[m][n]] += 1        
            
    # Turning Z into a matrix
    MaxLen = max([len(z[i]) for i in z.keys()])

    Z = np.zeros((len(z.keys()), MaxLen), dtype = int)

    for i in range(len(z.keys())):
        Z[i, 0:len(z[i])] = z[i]

    # Turning W into a matrix
    MaxLen = max([len(w[i]) for i in w.keys()])

    W = np.zeros((len(w.keys()), MaxLen), dtype = int)

    for i in range(len(w.keys())):
         W[i, 0:len(w[i])] = w[i]
    
    # set hyperparameters alpha and beta
    alpha = np.ones(K)
    beta = np.ones(V)
    
    
    
    return M, doc_lens, Z, W, K, A, B, C, alpha, beta

    
# start gibbs sampler
def Gibbs(M, doc_lens, Z, W, K, A, B, C, alpha, beta):
    """Basic Gibbs Sampler with now speedup"""
    num_iter = 1
    p = np.zeros(K)
    for i in range(num_iter):
        for m in range(M):
            for n in range(doc_lens[m]):
                A[m, int(Z[m,n])] -= 1  # Decrement N1
                B[int(Z[m,n]), int(W[m,n])] -= 1 # Decrement N2
                C[int(Z[m,n])] -= 1 # Decrement N3
                p = np.zeros(K)
                for k in range(K):
                    p[k] = (A[m, k] + alpha[k])*((B[k, int(W[m,n])] + beta[int(W[m,n])])/(C[k] + sum(beta)))
                p = p/sum(p) # This is actually doing k divisions... might be a modest speed up but we can parallelize easily with numba
                Z[m,n] = int(np.nonzero(np.random.multinomial(1, p))[0][0])
                A[m, int(Z[m,n])] += 1 # Increment N1
                B[int(Z[m,n]), int(W[m,n])] += 1 # Increment N2
                C[int(Z[m,n])] += 1 # Increment N3
    return A, B, C

# start gibbs sampler
def Gibbs_faster(M, doc_lens, Z, W, K, A, B, C, alpha, beta):
    """ Gibbs Sampler using optimization for Loop 4 """
    p = np.zeros(K)
    num_iter = 1
    for i in range(num_iter):
        for m in range(M):
            for n in range(doc_lens[m]):
                A[m, int(Z[m,n])] -= 1  # Decrement N1
                B[int(Z[m,n]), int(W[m,n])] -= 1 # Decrement N2
                C[int(Z[m,n])] -= 1 # Decrement N3
                p = Loop4(K, A, B, C, Z[m,n], W[m,n], alpha, beta, m)
                p = p/sum(p) 
                Z[m,n] = int(np.nonzero(np.random.multinomial(1, p))[0][0])
                A[m, int(Z[m,n])] += 1 # Increment N1
                B[int(Z[m,n]), int(W[m,n])] += 1 # Increment N2
                C[int(Z[m,n])] += 1 # Increment N3
    return A, B, C
    
    
def Gibbs_even_faster(M, doc_lens, Z, W, K, A, B, C, alpha, beta):
    """ Gibbs Sampler using optimization for Loop 3 and Loop 4"""
    p = np.zeros(K)
    num_iter = 1
    for i in range(num_iter):
        for m in range(M):
            A, B, C = Loop3(K, A, B, C, Z, W, alpha, beta, m, doc_lens[m])
    return A, B, C