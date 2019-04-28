# Load libraries
import numpy as np

def initialize(w, K, M, V, doc_lens):
    """
    Initializes values for collapsed Gibbs sampler.
    Returns topics and count matrices N_1, N_2, and N_3.
    """
    
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
    """
    Runs Gibbs sampler to get estimated latent topics.
    Returns count matrices N_1 and N_2.
    """
    
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
    """
    Calculates MC estimates for topic distributions using results from Gibbs sampler.
    Returns MxK matrix theta.
    """
    
    theta = np.zeros((M, K))
    for m in range(M):
        for k in range(K):
            theta[m, k] = (N_1[m, k] + alpha[k])/(doc_lens[m] + sum(alpha))
            
    return theta

def word_dist(N_2, beta, V, K):
    """
    Calculates MC estimates for word distributions using results from Gibbs sampler.
    Returns KxV matrix phi.
    """
    
    phi = np.zeros((K, V))
    for k in range(K):
        for v in range(V):
            phi[k, v] = (N_2[k, v] + beta[v]) / (sum(N_2[k, :]) + sum(beta))
            
    return phi

def lda(bow, K, alpha = 1, beta = 1, n_iter = 1000):
    """
    LDA implementation using collapsed Gibbs sampler.
    bow is a MxV bag-of-words matrix.
    alpha and beta are positive hyperparameters (either a signle value or list/array of length K).
    n_iter is the number of iterations for the sampler.
    Returns topic and word distributions.
    """
    
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
    if type(alpha) == int:
        alpha = np.ones(K) * alpha
        
    if type(beta) == int:
        beta  = np.ones(V) * beta
    
    # Run Gibbs sampler
    N_1, N_2 = gibbs(w, K, M, V, doc_lens, alpha, beta, N_1, N_2, N_3, z, n_iter)
    
    # Estimate topic and word distributions
    theta = topic_dist(N_1, doc_lens, alpha, M, K)
    phi   = word_dist(N_2, beta, V, K)
    
    return((theta, phi))

def group_docs(theta, K):
    """
    Uses LDA results to print most dominant topics in each document.
    """
    
    maxs = np.argmax(theta, axis = 1)
    for k in range(K):
        print("Documents labeled in group", k + 1, ":", np.where(maxs == k)[0])


def get_key_words(phi, n_words, words = None):
    """Uses LDA results to print key words from each topic."""
    K = len(phi)
    for k in range(K):
        biggest_probs = sorted(phi[k, :], reverse = False)[:n_words]
        key_words = [i for i in range(len(phi[k, :])) if phi[k, i] in biggest_probs]
        if words is not None:
            print("Key words for topic", k + 1, ": ", [words[i] for i in key_words])
        else:
            print("Key words for topic", k + 1, ": ", key_words)
