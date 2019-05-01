# Load libraries
import numpy as np
from numba import jit

@jit(nopython=True, cache = False)
def Loop4(K, A, B, C, WordTopic, TopicWord, alpha, beta, m):
    """ Optimized Version of Innermost loop of Gibbs sampler """
    p = np.zeros(K)
    for k in range(K):
        p[k] = (A[m, k] + alpha[k])*((B[k, TopicWord] + beta[TopicWord])/(C[k] + (beta[1]*len(beta))))
    return p


def get_multinom(p):
    """Multinomial Helper function"""
    p_sum = 0
    u = np.random.uniform(0,1)

    for i in range(len(p)):
        p_sum += p[i]
        if p_sum > u:
            return i
            break
             
def Loop3(K, A, B, C, Z, W, alpha, beta, m, doc_len):
    """ Optimized Loop 3 of Gibbs sampler """
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
