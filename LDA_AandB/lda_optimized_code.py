# Load libraries
import numpy as np
from numba import jit
import numba

@jit(nopython=True, cache = False)


def Loop4(K, A, B, C, WordTopic, TopicWord, alpha, beta, m):
    """ Function for the innermost loop: iterate over the topics """
    p = np.zeros(K)
    for k in range(K):
        p[k] = (A[m, k] + alpha[k])*((B[k, TopicWord] + beta[TopicWord])/(C[k] + (beta[1]*len(beta))))
    return p            

def get_multinom(p):
    """Helper function to generate a draw from multinomial distribution"""
    p_sum = 0
    u = np.random.uniform(0,1)

    for i in range(len(p)):
        p_sum += p[i]
        if p_sum > u:
            return i
            break

def Loop3(K, A, B, C, Z, W, alpha, beta, m, doc_len):
    """ Function for Loop 3: iterate over the documents"""
    p = np.zeros(K)
    for n in range(doc_len):
        WordTopic = Z[m,n]
        TopicWord = W[m,n]
        A[m, WordTopic] -= 1  # Decrement N1
        B[WordTopic, TopicWord] -= 1 # Decrement N2
        C[WordTopic] -= 1

        p = Loop4(K, A, B, C, WordTopic, TopicWord, alpha, beta, m)
        p = p/(np.sum(p))
        
        Z[m,n] = get_multinom(p)
        WordTopic = Z[m,n]
        A[m, WordTopic] += 1 # Increment N1
        B[WordTopic, TopicWord] += 1 # Increment N2
        C[WordTopic] += 1 # Increment N3
    return A, B, C

def Fastest_Gibbs(W, K, M, V, doc_lens, alpha, beta, A, B, C, Z, num_iter):
    """ Run Gibbs sampling iterations over the M documents"""
    for i in range(num_iter):
        for m in range(M):
            A, B, C = Loop3(K, A, B, C, Z, W, alpha, beta, m, doc_lens[m])
    return A, B, C


def Full_Gibbs(W, K, M, V, doc_lens, alpha, beta, A, B, C, Z, num_iter):
    """ Gibbs sampling with interior functions removed """
    p = np.zeros(K)
    for i in range(num_iter):
        for m in range(M):
            for n in range(doc_lens[m]):
                WordTopic = Z[m,n]
                TopicWord = W[m,n]
                A[m, WordTopic] -= 1  # Decrement N1
                B[WordTopic, TopicWord] -= 1 # Decrement N2
                C[WordTopic] -= 1
                
                for k in range(K):
                    p[k] = (A[m, k] + alpha[k])*((B[k, TopicWord] + beta[TopicWord])/(C[k] + (beta[1]*len(beta))))
                p = p/(np.sum(p))
                
                Z[m,n] = get_multinom(p)
                WordTopic = Z[m,n]
                A[m, WordTopic] += 1 # Increment N1
                B[WordTopic, TopicWord] += 1 # Increment N2
                C[WordTopic] += 1 # Increment N3
    return A, B, C

def initialize(W, K, M, V, doc_lens):
    """Initializes values for collapsed gibbs sampler"""
    
    # Set initial z randomly
    Z = np.zeros((M, K))
    for m in range(M):
        for n in range(doc_lens[m]):
            p = np.ones(K)/K
            Z[m,n] = get_multinom(p)
    
    # Create count matrices
    N_1 = np.zeros((M, K))
    for m in range(M):
        for k in range(K):
            N_1[m, k] = np.sum(np.array(Z[m,:]) == k)
            
    N_2 = np.zeros((K, V))
    for m in range(M):
        for n in range(doc_lens[m]):
            WordTopic = Z[m,n]
            TopicWord = W[m,n]
            N_2[WordTopic, TopicWord] += 1
            
    N_3 = np.zeros(K)
    for m in range(M):
        for n in range(doc_lens[m]):
            WordTopic = Z[m,n]
            N_3[Z[m,n]] += 1
            
    return((Z, N_1, N_2, N_3))

def topic_dist(N_1, doc_lens, alpha, M, K):
    """Calculates MC estimates for topic distributions using results from Gibbs sampler"""
    
    theta = np.zeros((M, K))
    for m in range(M):
        for k in range(K):
            theta[m, k] = (N_1[m, k] + alpha[k])/(sum(N_1[k, :]) + sum(alpha))
            
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
    max_len = np.max(doc_lens)
    W = np.zeros((M, max_len))
    
    for m in range(M):
        d = 0
        for v in range(V):
            for n in range(int(bow[m, v])):
                W[m,d] = v
                d+=1
    
    # Initialize values for Gibbs sampler   
    Z, N_1, N_2, N_3 = initialize(W, K, M, V, doc_lens)
    
    
    # Set symmetric hyperparameters
    alpha = np.ones(K) * alpha
    beta  = np.ones(V) * beta
        
    
    # Run Gibbs sampler
    N_1, N_2 = Fastest_Gibbs(W, K, M, V, doc_lens, alpha, beta, N_1, N_2, N_3, Z, n_iter)
    
    # Estimate topic and word distributions
    theta = topic_dist(N_1, doc_lens, alpha, M, K)
    phi   = word_dist(N_2, beta, V, K)
    
    return((theta, phi))
