# -*- coding: utf-8 -*-
import gensim
import nltk
import numpy as np
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
import math
import build_source_file
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

stopWords = stopwords.words('english')
stopWords += ['.']


def idf_dict(file_dir):
    files = build_source_file.find_all_files(file_dir)
    documents = []
    vocabulary = set()
    for file in files:
        with open(file, 'r') as f:
            line = f.read().split('¥n')
            re_line = build_source_file.reform_text(line)
            re_token, re_line = tokenize(re_line, rm_stop=True)

            extended_token = []
            for tokens in re_token:
                extended_token.extend(tokens)

            documents.append(set(extended_token))

    for document in documents:
        vocabulary = vocabulary | document

    N = float(len(documents))
    word_to_idf = {}
    print N
    

    for word in vocabulary:
        cnt = 0
        for document in documents:
            if word in document:
                cnt += 1

        word_to_idf[word] = math.log(N / cnt) + 1

    return word_to_idf




def tokenize(docs, rm_stop=False):
    """

    :param docs: a list of sentence(strings)
    :return:
    """

    token_list = []
    re_document = []

    for doc in docs:
        sentence = doc
        tokens = nltk.word_tokenize(sentence.lower())
        if rm_stop:
            tokens = rm_stopwords2(tokens)

        if len(tokens) != 0:
            token_list.append(nltk.word_tokenize(doc.lower()))
            re_document.append(sentence)


    return token_list, re_document

def rm_stopwords(token_list):
    token_list_stop = []
    for tokens in token_list:
        tokens_stop = []
        for token in tokens:
            if token not in stopWords:
                tokens_stop.append(token)

        token_list_stop.append(tokens_stop)

    return token_list_stop

def rm_stopwords2(token_list):
    token_list_stop = []
    for token in token_list:
        if token not in stopWords:
            token_list_stop.append(token)

    return token_list_stop

def get_wordmatrix(medical=False):
    """

    :return: dict of word to id
            list of word vectors (np array)
    """
    if medical:
        model = gensim.models.Word2Vec.load("/home/suzukilab/Documents/word2vec/trunk/PMC_open_access.model")
    else:
        model = gensim.models.KeyedVectors.load_word2vec_format("/home/suzukilab/Documents/word2vec/trunk/GoogleNews-vectors-negative300.bin", binary=True)

    word_to_id = {}
    word_vectors = []

    for idx, word in enumerate(model.wv.vocab):
        word_to_id[word] = idx
        word_vectors.append(model.wv[word])

    word_vectors = np.array(word_vectors)

    del model

    return word_to_id, word_vectors

def sentence_length(doc):
    """

    :param doc: a list of sentence (a list of words)
    :return: np array
    """
    length = []

    for sentence in doc:
        length.append(len(sentence))

    length = np.array(length)

    return length


def get_phrase_vector_tfidf(docs, word_to_id=None, word_vectors=None, word_to_idf=None):
    """
    calc document vector
    :param doc: list of pretrained document(word token list)
    :param word_to_id: dict of word to id
    :param word_vectors: list of word vectors (np array)
    :return: sentence vectors in all documents
    """
    
    extended_doc = []
    for doc in docs:
        extended_doc.extend(doc)

    N = float(len(extended_doc))
    
    word_to_tf ={}
    for token in extended_doc:
        word_to_tf.setdefault(token, 0)
        word_to_tf[token] += 1

    phrase_vectors = []

    for doc in docs:
        phrase_vector = np.zeros(word_vectors.shape[1])
        for term in doc:
            if term in word_to_id:
                if term not in word_to_idf:
                    word_to_idf[term] = 1
                phrase_vector += (word_to_tf[term] / N) * word_to_idf[term] * word_vectors[word_to_id[term]]

        phrase_vector = phrase_vector / len(doc)
        phrase_vectors.append(phrase_vector)

    phrase_vectors = np.array(phrase_vectors)
    return phrase_vectors

def get_phrase_vector_add(docs, word_to_id, word_vectors):
    """
    calc document vector
    :param docs: list of pre-trained document(word token list)
    :param word_to_id: dict of word to id
    :param word_vectors: list of word vectors (np array)
    :return: sentence vectors in all documents
    """

    phrase_vectors = []

    for doc in docs:
        phrase_vector = np.zeros(word_vectors.shape[1])
        for term in doc:
            if term in word_to_id:
                phrase_vector += word_vectors[word_to_id[term]]

        phrase_vector = phrase_vector / len(doc)
        phrase_vectors.append(phrase_vector)


    return phrase_vectors

def sim_euclid(doc_vecs):
    """

    :param doc_vecs: list of sentence vectors in all documents(np array)
    :return:
    """

    sim_matrix = np.zeros((len(doc_vecs), len(doc_vecs)))

    for i in range(len(doc_vecs)):
        for j in range(len(doc_vecs)):
            sim_matrix[i][j] = np.linalg.norm(doc_vecs[i] - doc_vecs[j])

    sim_matrix = sim_matrix / sim_matrix.max()
    sim_matrix = np.ones_like(sim_matrix) - sim_matrix

    return sim_matrix

def sentence_value(sim_matrix):
    """

    :param sim_matrix: similarity of all sentence(np array)
    :return:
    """

    value = (sim_matrix.sum(axis=1) - 1) / (sim_matrix.shape[1] - 1)

    return value

def clustring(doc_vecs, cluster_num = 4):
    """
    k means clustering
    :param doc_vecs: sentence vectors in all documents(np array)
    :return:
    """

    pred = KMeans(n_clusters=cluster_num).fit_predict(doc_vecs)
    pred = np.array(pred)
    return pred

def diversity_function(summary_idx, value, cluster_list):
    """

    :param summary_idx: summary index
    :param value: sentence value of all sentence
    :param cluster_list: sentence cluster of all sentence
    :return: diversity score
    """
    cluster_num = cluster_list.max() + 1
    diversities = np.zeros(cluster_num)

    i = 0
    while i != len(summary_idx):
        diversities[cluster_list[summary_idx[i]]] += value[summary_idx[i]]
        i += 1

    diversities = np.sqrt(diversities)
    diversities = np.sum(diversities)

    return diversities

def coverage_function(summary_idx, sim_matrix, alpha):
    """

    :param summary_idx: sentence index of summary sentence
    :param sim_matrix: similarity matrix
    :return: coverage score
    """

    coverage = 0
    doc_num = sim_matrix.shape[0]
    summary_num = len(summary_idx)
    i = 0
    while i != doc_num:
        summary_coverage = 0
        j = 0
        while j != summary_num:
            summary_coverage += sim_matrix[i, summary_idx[j]]
            j += 1

        total_coverage = 0
        j = 0
        while j != doc_num:
            total_coverage += sim_matrix[i, j]
            j += 1

        coverage += min(summary_coverage, alpha * total_coverage)
        i += 1


    return coverage

def submodular_function(value, cluster_list, summary_idx, sim_matrix, alpha, lamb):
    coverage = coverage_function(summary_idx, sim_matrix, alpha)
    diversity = diversity_function(summary_idx, value, cluster_list)

    return (coverage + lamb * diversity)


def modified_greedy_algorithm(sentence_length, sentence_value, cluster_list, sim_matrix, alpha, lamb, abstract_idx, r, max_length):
    """

    :param sentence_length: a list of sentence length in all documents(np array)
    :param sentences_value: a list of sentence value in all documents(np array)
    :param cluster_list: a list of cluster in all documents(np array)
    :param sim_matrix: similarity matrix of all documents (np array)
    :param abstract_idx: a list of index of abstract
    :param r: scaling function
    :param max_length: max words length in summary
    :return:
    """
    g = []
    u = abstract_idx

    while u != []:
        value = np.zeros(len(u))
        i = 0
        while i != value.shape[0]:
            value[i] = submodular_function(sentence_value, cluster_list, g + [u[i]], sim_matrix, alpha, lamb) - submodular_function(sentence_value, cluster_list, g, sim_matrix, alpha, lamb)
            value[i] = value[i] / np.power(sentence_length[u[i]], r)

            i += 1


        s = np.argmax(value)

        length = 0
        for idx in g + [u[s]]:
            length += sentence_length[idx]

        if length <= max_length:
            g += [u[s]]

        u = np.delete(u, s)

    value = np.zeros(len(abstract_idx))
    i = 0
    while i != value.shape[0]:
        value[i] = submodular_function(sentence_value, cluster_list, [abstract_idx[i]], sim_matrix, alpha, lamb)
        i += 1

    s = abstract_idx[np.argmax(value)]

    if submodular_function(sentence_value, cluster_list, g, sim_matrix, alpha, lamb) > submodular_function(sentence_value, cluster_list, [s], sim_matrix, alpha, lamb):
        g_total = g
    else:
        g_total = [s]

    return g_total


def decomposition(vecs, n=2):
    pca = PCA(n_components=n)
    pca.fit(vecs)
    data_pca = pca.transform(vecs)

    return data_pca

def plot(documents_vecs, documents_annotates, summary_vecs, summary_annotation):

    for i in range(len(documents_vecs)):
        plt.plot(documents_vecs[i][0], documents_vecs[i][1], 'o', color='b')
        #plt.annotate(documents_annotates[i], (documents_vecs[i][0], documents_vecs[i][1]))

    for i in range(len(summary_vecs)):
        plt.plot(summary_vecs[i][0], summary_vecs[i][1], 'o', color='r')

    plt.show()

"""
docs_string = ["It will be sunny tomorrow.", "It is rainy today.", "I play tennis because it is cloudy."]
N = len(docs_string)
tokens = tokenize(docs_string)
docs = rm_stopwords(tokens)
word_to_id, word_vectors = get_wordmatrix()
phrase_vectors = get_phrase_vector_tfidf(docs, word_to_id, word_vectors)
sim_matrix = sim_euclid(phrase_vectors)
sentence_length = sentence_length(tokens)
value = sentence_value(sim_matrix)
cluster_list = clustring(phrase_vectors, cluster_num=2)

g = modified_greedy_algorithm(sentence_length, value, cluster_list, sim_matrix, 0.1, 4, [0, 1, 2], 0.3, 15)
for idx in g:
    print(docs_string[idx])
"""
