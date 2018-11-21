# -*- coding: utf-8 -*-

documents_path = '/media/suzukilab/UbuntuData/corpus/obesity/body'
reference_path = '/media/suzukilab/UbuntuData/corpus/obesity/summary'

output_root_dir = ''


import build_source_file
import utility_function
import multsum_preprocess
import numpy as np
import multsum
from pythonrouge.pythonrouge import Pythonrouge
import wv_opt
from sklearn.model_selection import KFold


def load_corpus(doc_path, ref_path):
    documents = []
    summary = []
    
    index_to_filename = {}

    document_files = build_source_file.find_all_files(doc_path)

    cnt = 0
    for file in document_files:
        article_id = file.split('/')[-1].split('.')[0]
        summary_file = ref_path + '/' + article_id + '_summary.txt'

        with open(file, 'r') as f:
            body = f.read()

        with open(summary_file, 'r') as f:
            abstract = f.read()

        abstract = abstract.split('\n')
        body = body.split('\n')

        abstract = build_source_file.reform_text(abstract)
        body = build_source_file.reform_text(body)



        abstract_num = 0
        for abst in abstract:
            abstract_num += len(abst.split(' '))

        if abstract != [] and (abstract_num >= 100) and (len(body) >= 50):
            index_to_filename[len(documents)] = article_id
            documents.append(body)
            summary.append(abstract)
            cnt += 1


    print len(documents)
    return documents, summary, index_to_filename

def preprocess_document(documents):
    preprocessed_documents = []
    preprocessed_documents_tokens = []

    for document in documents:
        re_document = build_source_file.reform_text(document)
        re_token, re_document = utility_function.tokenize(re_document, rm_stop=True)
        preprocessed_documents.append(re_document)
        preprocessed_documents_tokens.append(re_token)

    preprocessed_documents_tokens = multsum_preprocess.preprocess(preprocessed_documents_tokens)

    return preprocessed_documents, preprocessed_documents_tokens


def create_document_vecs(docs, word_to_id, word_vectors):
    vecs = []
    for doc in docs:
        documents_vec = utility_function.get_phrase_vector_add(doc, word_to_id, word_vectors)
        vecs.append(documents_vec)

    return vecs

def create_document_vecs_tfidf(docs, word_to_id, word_vectors, word_to_idf):
    vecs = []
    for doc in docs:
        documents_vec = utility_function.get_phrase_vector_tfidf(doc, word_to_id, word_vectors, word_to_idf)
        vecs.append(documents_vec)
    
    return vecs

def transform_vecs(vecs):
    trans_vecs_list = []
    """
    for doc in vecs:
        trans_vecs = []
        for vec in doc:
            trans_vec = np.c_[vec]
            trans_vecs.append(trans_vec)
        trans_vecs_list.append(trans_vecs)
    """
    for vec in vecs:
        trans_vec = np.c_[vec]
        trans_vecs_list.append(trans_vec)


    return trans_vecs_list

def summarize_kobayashi(documents_token, documents, word_to_id, word_vectors, clustering_wv, tfidf, word_to_idf):
    if tfidf:
        documents_vecs = create_document_vecs_tfidf(documents_token, word_to_id, word_vectors, word_to_idf)
        cluster_documents_vecs = create_document_vecs_tfidf(documents_token, word_to_id, clustering_wv, word_to_idf)
    else:
        documents_vecs = create_document_vecs(documents_token, word_to_id, word_vectors)
        cluster_documents_vecs = create_document_vecs(documents_token, word_to_id, clustering_wv)
    
    summary_list = []
    for documents_vec,cluster_documents_vec, document in zip(documents_vecs, cluster_documents_vecs, documents):
        documents_num = len(documents_vec)
        sim_matrix = utility_function.sim_euclid(documents_vec)
        cluster_sim_matrix = utility_function.sim_euclid(cluster_documents_vec)
        summary = multsum.summarize_matrix_files_kobayashi(matrix=[sim_matrix],
                                                           clustering_matrix=cluster_sim_matrix,
                                                           documents=[document],
                                                           length=int(documents_num*0.3),
                                                           unit=2,
                                                           output_numbers=False,
                                                           quiet=True,
                                                           sentenceVectors=documents_vec)
                                                   
        summary_list.append(summary)
    
    return summary_list

def summarize(documents_token, documents, word_to_id, word_vectors, clustering_wv, tfidf, word_to_idf):
    if tfidf:
        documents_vecs = create_document_vecs_tfidf(documents_token, word_to_id, word_vectors, word_to_idf)
        cluster_documents_vecs = create_document_vecs_tfidf(documents_token, word_to_id, clustering_wv, word_to_idf)
    else:
        documents_vecs = create_document_vecs(documents_token, word_to_id, word_vectors)
        cluster_documents_vecs = create_document_vecs(documents_token, word_to_id, clustering_wv)

    summary_list = []
    for documents_vec,cluster_documents_vec, document in zip(documents_vecs, cluster_documents_vecs, documents):
        documents_num = len(documents_vec)
        sim_matrix = utility_function.sim_euclid(documents_vec)
        cluster_sim_matrix = utility_function.sim_euclid(cluster_documents_vec)

        summary = multsum.summarize_matrix_files_u(matrix=[sim_matrix],
                                                   clustering_matrix=cluster_sim_matrix,
                                                   documents=[document],
                                                   length=int(documents_num*0.3),
                                                   unit=2,
                                                   output_numbers=False,
                                                   quiet=True)

        summary_list.append(summary)

    return summary_list

def opt_vecs(documents_token, references_token, word_to_id, word_vectors, tfidf, word_to_idf):
    if tfidf:
        documents_vecs = create_document_vecs_tfidf(documents_token, word_to_id, word_vectors, word_to_idf)
        references_vecs = create_document_vecs_tfidf(references_token, word_to_id, word_vectors, word_to_idf)
    else:
        documents_vecs = create_document_vecs(documents_token, word_to_id, word_vectors)
        references_vecs = create_document_vecs(references_token, word_to_id, word_vectors)

    documents_sum_vecs = []
    for doc_vecs in documents_vecs:
        doc_sum_vec = np.zeros_like(doc_vecs[0], dtype=float)
        for doc_vec in doc_vecs:
            doc_sum_vec += doc_vec

        doc_sum_vec = doc_sum_vec / len(doc_vecs)

        documents_sum_vecs.append(doc_sum_vec)

    documents_vecs = transform_vecs(documents_sum_vecs)

    invert_ref = []
    for reference_vec in references_vecs:
        invert_ref.append(transform_vecs(reference_vec))

    return documents_vecs, invert_ref

def opt_vecs_taguchi(documents_token, references_token, word_to_id, word_vectors, tfidf, word_to_idf):
    if tfidf:
        documents_vecs = create_document_vecs_tfidf(documents_token, word_to_id, word_vectors, word_to_idf)
        references_vecs = create_document_vecs_tfidf(references_token, word_to_id, word_vectors, word_to_idf)
    else:
        documents_vecs = create_document_vecs(documents_token, word_to_id, word_vectors)
        references_vecs = create_document_vecs(references_token, word_to_id, word_vectors)

    documents_sum_vecs = []
    for doc_vecs in documents_vecs:
        doc_sum_vec = np.zeros_like(doc_vecs[0], dtype=float)
        for doc_vec in doc_vecs:
            doc_sum_vec += doc_vec
    
        doc_sum_vec = doc_sum_vec / len(doc_vecs)

        documents_sum_vecs.append(doc_sum_vec)

    documents_vecs = transform_vecs(documents_sum_vecs)

    references_sum_vecs = []
    for ref_vecs in references_vecs:
        ref_sum_vec = np.zeros_like(ref_vecs[0], dtype=float)
        for ref_vec in ref_vecs:
            ref_sum_vec += ref_vec
        
        ref_sum_vec = ref_sum_vec / len(ref_vecs)
        
        references_sum_vecs.append(ref_sum_vec)
    

    references_vecs = transform_vecs(references_sum_vecs)

    return documents_vecs, references_vecs

def get_new_wordvector(A, W):
    return (A.dot(W.T)).T


def summarization_model_kobayashi(train_documents_token, train_references_token, test_documents_token, test_documents, test_references, word_to_id, word_vectors, opt=True, opt_lambda=0.05, tfidf=False, word_to_idf=None):
    if opt:
        opt_doc, opt_ref = opt_vecs_taguchi(train_documents_token, train_references_token, word_to_id, word_vectors, tfidf, word_to_idf)
        wv = wv_opt.WV_opt(opt_lambda, word_vectors)
        A = wv.fit_inverse_taguchi(opt_ref, opt_doc)
        
        opt_word_vectors = get_new_wordvector(A, word_vectors)
        
        summary_list = summarize_kobayashi(test_documents_token, test_documents, word_to_id, opt_word_vectors, opt_word_vectors, tfidf, word_to_idf)
    else:
        summary_list = summarize_kobayashi(test_documents_token, test_documents, word_to_id, word_vectors, word_vectors, tfidf, word_to_idf)
    
    return summary_list

def summarization_model(train_documents_token, train_references_token, test_documents_token, test_documents, test_references, word_to_id, word_vectors, opt=True, opt_lambda=0.05, tfidf=False, word_to_idf=None):
    if opt:
        opt_doc, opt_ref = opt_vecs(train_documents_token, train_references_token, word_to_id, word_vectors, tfidf, word_to_idf)
        wv = wv_opt.WV_opt(opt_lambda, word_vectors)
        A = wv.fit_inverse(opt_ref, opt_doc)

        opt_word_vectors = get_new_wordvector(A, word_vectors)

        summary_list = summarize(test_documents_token, test_documents, word_to_id, opt_word_vectors, word_vectors, tfidf, word_to_idf)
    else:
        summary_list = summarize(test_documents_token, test_documents, word_to_id, word_vectors, word_vectors, tfidf, word_to_idf)

    return summary_list


def culc_rouge(summary_path, reference_path):
    files = build_source_file.find_all_files(summary_path)
    summary = []
    references = []
    for file in files:
        id = file.split('/')[-1].split('.')[0]
        with open(file, 'r') as f:
            lines = f.read().split('\n')
            summary.append(lines)

        with open(reference_path + '/' + id + '.1.txt', 'r') as f:
            lines = f.read().split('\n')
            references.append([lines])

    rouge = Pythonrouge(summary_file_exist=False,
                        summary=summary, reference=references,
                        n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
                        recall_only=False, stemming=True, stopwords=False,
                        word_level=True, length_limit=False, length=500,
                        use_cf=True, cf=95, scoring_formula='average',
                        resampling=True, samples=1000, favor=True, p=0.5)
    score = rouge.calc_score()

    return score

def rouge_score_to_file(score, rouge_output_path):
    with open(rouge_output_path, 'w') as f:
        for method, val in score.iteritems():
            f.write(method + ':' + str(val) + '\n')


documents, references, index_to_filename = load_corpus(documents_path, reference_path)

word_to_id, word_vectors = utility_function.get_wordmatrix(medical=True)

#word_to_idf = utility_function.idf_dict(documents_path)
print "prephase is done"
rouge1_score = []
rouge2_score = []
kf = KFold(n_splits=5)
cnt = 0
for train_index, test_index in kf.split(documents):
    cnt += 1
    print cnt
    train_documents = []
    train_references = []
    test_documents = []
    test_references = []
    index_to_test_output_path ={}
    
    for i in train_index:
        train_documents.append(documents[i])
        train_references.append(references[i])
    
    for i in test_index:
        index_to_test_output_path[len(test_references)] = index_to_filename[i]
        test_documents.append(documents[i])
        test_references.append(references[i])
        
    pre_train_documents, train_documents_token = preprocess_document(train_documents)
    pre_train_references, train_references_token = preprocess_document(train_references)
    pre_test_documents, test_documents_token = preprocess_document(test_documents)



    output_root_dir = '/media/suzukilab/UbuntuData/corpus/obesity/shogo_default_optimized_0.00001'
    summary_list = summarization_model(train_documents_token, train_references_token, test_documents_token, pre_test_documents, test_references, word_to_id, word_vectors, opt=True, opt_lambda=0.00001, tfidf=False, word_to_idf=None)
    
    for i in range(len(test_references)):
        with open(output_root_dir + '/summary/' + index_to_test_output_path[i] + '.txt', 'w') as f:
            f.write('\n'.join(summary_list[i]))
        with open(output_root_dir + '/reference/' + index_to_test_output_path[i] + '.1.txt', 'w') as f:
            f.write('\n'.join(test_references[i]))




rouge_summary_path = '/media/suzukilab/UbuntuData/corpus/obesity/shogo_default_optimized_0.00001/summary'
rouge_reference_path = '/media/suzukilab/UbuntuData/corpus/obesity/shogo_default_optimized_0.00001/reference'
rouge_output_path = '/media/suzukilab/UbuntuData/corpus/obesity/shogo_default_optimized_0.00001/result/rouge_score.txt'
rouge_score = culc_rouge(rouge_summary_path, rouge_reference_path)
rouge_score_to_file(rouge_score, rouge_output_path)





