import build_source_file
import utility_function
import numpy as np
import summary
import wv_opt

documents_path = '/media/suzukilab/UbuntuData/corpus/obesity/body'
reference_path = '/media/suzukilab/UbuntuData/corpus/obesity/summary'

documents, references, index_to_filename = summary.load_corpus(documents_path, reference_path)

word_to_id, word_vectors = utility_function.get_wordmatrix(medical=True)

pre_documents, documents_token = summary.preprocess_document(documents)
pre_references, references_token = summary.preprocess_document(references)

documents_vec = summary.create_document_vecs(documents_token, word_to_id, word_vectors)
summary_vec = summary.create_document_vecs(references_token, word_to_id, word_vectors)

opt_doc, opt_ref = summary.opt_vecs(documents_token[100:], references_token[100:], word_to_id, word_vectors, tfidf=False, word_to_idf=None)
wv = wv_opt.WV_opt(0.00001, word_vectors)
A = wv.fit_inverse(opt_ref, opt_doc)
opt_word_vectors = summary.get_new_wordvector(A, word_vectors)


opt_documents_vec = summary.create_document_vecs(documents_token, word_to_id, opt_word_vectors)
opt_summary_vec = summary.create_document_vecs(references_token, word_to_id, opt_word_vectors)
dec_vec = utility_function.decomposition(documents_vec[0])
dec_summary_vec = utility_function.decomposition(summary_vec[0])
utility_function.plot(dec_vec, pre_documents[0], dec_summary_vec, pre_references[0])

opt_dec_vec = utility_function.decomposition(opt_documents_vec[0])
opt_dec_summary_vec = utility_function.decomposition(opt_summary_vec[0])
utility_function.plot(opt_dec_vec, pre_documents[0], opt_dec_summary_vec, pre_references[0])

