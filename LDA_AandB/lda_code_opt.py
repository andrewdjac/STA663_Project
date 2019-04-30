# Load libraries
import numpy as np
from LDA_AandB.lda_optimized_code import Full_Gibbs, get_multinom
from LDA_AandB.lda_code import topic_dist, word_dist

def initialize(W, K, M, V, doc_lens):
    """Initializes values for collapsed gibbs sampler"""
    
    # Set initial z randomly
    max_len = np.max(doc_lens)
    Z = np.zeros((M, max_len), dtype = 'int')
    for m in range(M):
        for n in range(doc_lens[m]):
            Z[m,n] = get_multinom(np.ones(K)/K)
    
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

def lda_optimized(bow, K, alpha = 1, beta = 1, n_iter = 1000):
    """LDA implementation using collapsed Gibbs sampler"""
    
    # Get corpus parameters
    M, V = bow.shape
    doc_lens = np.sum(bow, axis = 1, dtype = 'int')
    
    # Create word dictionary
    max_len = np.max(doc_lens)
    W = np.zeros((M, max_len), dtype='int')
    
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
    A, B, C= Full_Gibbs(W, K, M, V, doc_lens, alpha, beta, N_1, N_2, N_3, Z, n_iter)
    
    
    # Estimate topic and word distributions
    theta = topic_dist(A, doc_lens, alpha, M, K)
    phi   = word_dist(B, beta, V, K)
    return((theta, phi))
