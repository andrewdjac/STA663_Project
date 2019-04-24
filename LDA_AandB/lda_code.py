# Load libraries
import numpy as np

def initialize(w, K, M, V, doc_lens):
    """Initializes values for collapsed gibbs sampler"""
    
    # Set initial z randomly
    z = {}
    for m in range(M):
        z[m] = []
        for n in range(doc_lens[m]):
            z[m].extend(np.nonzero(np.random.multinomial(1, np.ones(K)/K))[0])
    
    # Create count matrices
    N_1 = np.zeros((M, K))
    for m in range(M):
        for k in range(K):
            N_1[m, k] = sum(np.array(z[m]) == k)
            
    N_2 = np.zeros((K, V))
    for m in range(M):
        for n in range(doc_lens[m]):
            N_2[z[m][n], w[m][n]] += 1
            
    N_3 = np.zeros(K)
    for m in range(M):
        for n in range(doc_lens[m]):
            N_3[z[m][n]] += 1
            
    return((z, N_1, N_2, N_3))

def gibbs(w, K, M, V, doc_lens, alpha, beta, N_1, N_2, N_3, z, n_iter):
    """Runs gibbs sampler to get estimated latent topics"""
    
    for i in range(n_iter):
        for m in range(M):
            for n in range(doc_lens[m]):
                N_1[m, z[m][n]] -= 1
                N_2[z[m][n], w[m][n]] -= 1
                N_3[z[m][n]] -= 1
                p = np.zeros(K)
                for k in range(K):
                    p[k] = (N_1[m, k] + alpha[k])*((N_2[k, w[m][n]] + beta[w[m][n]])/(N_3[k] + sum(beta)))
                p /= sum(p)
                z[m][n] = np.nonzero(np.random.multinomial(1, p))[0][0]
                N_1[m, z[m][n]] += 1
                N_2[z[m][n], w[m][n]] += 1
                N_3[z[m][n]] += 1
                
    return((N_1, N_2))

def topic_dist(N_1, doc_lens, alpha, M, K):
    """Calculates MC estimates for topic distributions using results from Gibbs sampler"""
    
    theta = np.zeros((M, K))
    for m in range(M):
        for k in range(K):
            theta[m, k] = (N_1[m, k] + alpha[k])/(doc_lens[m] + sum(alpha))
            
    return theta

def word_dist(N_2, beta, V, K):
    """Calculates MC estimates for word distributions using results from Gibbs sampler"""
    
    phi = np.zeros((K, V))
    for k in range(K):
        for v in range(V):
            phi[k, v] = (N_2[k, v] + beta[v]) / (sum(N_2[k, :]) + sum(beta))
            
    return phi

def lda(bow, K, alpha = 1, beta = 1, n_iter = 1000):
    """LDA implementation using collapsed Gibbs sampler"""
    
    # Get corpus parameters
    M, V = bow.shape
    doc_lens = np.sum(bow, axis = 1, dtype = 'int')
    
    # Create word dictionary
    w = {}
    for m in range(M):
        w[m] = []
        for v in range(V):
            for n in range(int(bow[m, v])):
                w[m].append(v)
    
    # Initialize values for Gibbs sampler   
    z, N_1, N_2, N_3 = initialize(w, K, M, V, doc_lens)
    
    
    # Set symmetric hyperparameters
    alpha = np.ones(K) * alpha
    beta  = np.ones(V) * beta
    
    # Run Gibbs sampler
    N_1, N_2 = gibbs(w, K, M, V, doc_lens, alpha, beta, N_1, N_2, N_3, z, n_iter)
    
    # Estimate topic and word distributions
    theta = topic_dist(N_1, doc_lens, alpha, M, K)
    phi   = word_dist(N_2, beta, V, K)
    
    return((theta, phi))

def group_docs(theta, K):
    """Uses LDA results to give most dominant topics in each document"""
    
    maxs = np.argmax(theta, axis = 1)
    for k in range(K):
        print("Documents labeled in group", k + 1, ":", np.where(maxs == k)[0])
