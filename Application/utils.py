import pandas as pd
from collections import defaultdict, Counter
import re
import math
from nltk.tokenize import word_tokenize

def clean_text(text):
    text = re.sub(r'\W+', ' ', text)
    return text.lower()

def build_inverted_index(docs):
    inverted_index = defaultdict(list)
    for doc_id, doc_fields in enumerate(docs):
        combined_fields = " ".join([str(field) for field in doc_fields]) 
        tokens = word_tokenize(clean_text(combined_fields))
        for token in tokens:
            if doc_id not in inverted_index[token]:
                inverted_index[str(token)].append(doc_id)
    return inverted_index

def calculate_tf(doc):
    tokens = word_tokenize(clean_text(doc))
    total_tokens = len(tokens)
    tf = Counter(tokens)
    return {token: count / total_tokens for token, count in tf.items()}

def calculate_idf(documents, inverted_index):
    N = len(documents)
    idf = {}
    for token in inverted_index:
        df = len(inverted_index[token])
        idf[token] = math.log(N / (1 + df))
    return idf

def calculate_tfidf(documents, inverted_index):
    idf = calculate_idf(documents, inverted_index) 
    tfidf_documents = []
    
    for doc_fields in documents:
        combined_fields = " ".join([str(field) for field in doc_fields])
        tf = calculate_tf(combined_fields)
        tfidf = {token: tf[token] * idf.get(token, 0) for token in tf}
        tfidf_documents.append(tfidf)
    
    all_tokens = sorted(set(token for tfidf in tfidf_documents for token in tfidf))
    tfidf_df = pd.DataFrame(0.0, index=range(len(documents)), columns=all_tokens)
    
    for doc_id, tfidf in enumerate(tfidf_documents):
        for token, value in tfidf.items():
            tfidf_df.at[doc_id, token] = value
    
    return tfidf_df

def search_document(tfidf_df, search_term):
    search_term = search_term.lower()
    if search_term in tfidf_df.columns:
        ranked_docs = tfidf_df[search_term].sort_values(ascending=False)
        ranked_docs = ranked_docs[ranked_docs != 0]
        return ranked_docs
    else: return None
        
    
def return_found_information(original_df, ranked_docs, search_term):
    if ranked_docs is not None:
        indices = ranked_docs.index
        found_information = original_df.loc[indices]
        return found_information
    elif search_term == "": return search_term
    else: return(f"'{search_term}' not found in player data.")
