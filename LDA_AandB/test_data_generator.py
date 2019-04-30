# Load libraries
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix

def generate_dists(alpha, beta, M, K, V):
    """
    Generates topic and word distributions.
    alpha and beta are hyperparameters of length K.
    Returns MxK matrix theta and KxV matrix phi.
    """
    
    # Generate word distributions
    phi = np.zeros((K, V))
    for k in range(K):
        phi[k, :] = np.random.dirichlet(beta)
    
    # Generate topic distributions
    theta = np.zeros((M, K))
    for m in range(M):
        theta[m,:] = np.random.dirichlet(alpha)
    
    return (phi, theta)


def generate_words(phi, theta, M, N_min, N_max):
    """
    Generates 'words' for corpus.
    Return word dictionary w.
    """
    
    doc_lens = np.random.randint(N_min, N_max, M)
    z = {}
    w = {}
    for m in range(M):
        z[m] = []
        w[m] = []
        for n in range(doc_lens[m]):
            z[m].extend(np.nonzero(np.random.multinomial(1, theta[m,:]))[0])
            w[m].extend(np.nonzero(np.random.multinomial(1, phi[z[m][n], :]))[0])
    
    return w


def make_bow(w, M, V):
    """Creates bag-of-words matrix from corpus."""
    
    bow = np.zeros((M, V))
    for m in range(M):
        for v in range(V):
            bow[m, v] = len(np.where(np.array(w[m]) == v)[0])
    
    return bow


def simulate_corpus(alpha, beta, M, N_min, N_max):
    """Generates test data for LDA.
    Returns MxV bag-of-words matrix bow, 
    MxK topic distribution matrix theta, 
    and KxV word distribution matrix phi. 
    """
    
    # Get corpus parameters
    K = len(alpha)
    V = len(beta)
    
    # Generate topic and word distributions
    phi, theta = generate_dists(alpha, beta, M, K, V)
    
    # Generate words
    w = generate_words(phi, theta, M, N_min, N_max)
    
    # Make bag-of-words matrix
    bow = make_bow(w, M, V)
    
    return (bow, theta, phi)


newsgroups_categories = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']

def get_newsgroups(categories = None, n_articles = 10):
    """
    Fetches random newsgroups articles of specified categories.
    categories is a list of valid categories in Newsgroups.
    n_articles is the number of articles to extract.
    """
    
    remove = ('headers', 'footers', 'quotes')
    newsgroups = fetch_20newsgroups(subset = 'train', remove = remove, categories = categories)
    
    ind = np.random.choice(len(newsgroups.data), size = n_articles, replace = False)
    news = [newsgroups.data[i] for i in ind]
    labels = [newsgroups.target[i] for i in ind]
    
    words = [' '.join(filter(str.isalpha, raw.lower().split())) for raw in
        news]

    vectorizer = CountVectorizer()
    vectorizer.fit(words)
    wordbank = vectorizer.get_feature_names()
    
    bow_sparse = vectorizer.transform(words)
    bow = np.array(csr_matrix.todense(bow_sparse))
    
    return (bow, labels, wordbank)