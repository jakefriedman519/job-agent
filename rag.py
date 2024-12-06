# prompt: show me software engineering jobs

import math
from collections import defaultdict
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import nltk
from nltk.corpus import stopwords

import jobs
from jobs import job_listings_strings

# Ensure nltk stopwords are downloaded
nltk.download('stopwords')

# Initialize stopwords
stop_words = set(stopwords.words('english'))

# Load pre-trained BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Ensure the model is in evaluation mode
model.eval()


def clean_text(text: str) -> str:
    """
    Clean the text by removing stopwords and making the text lowercase.
    """
    words = text.lower().split()
    return " ".join([word for word in words if word not in stop_words])


def compute_tf_matrix(documents: List[str]) -> Tuple[np.ndarray, dict]:
    """
    Compute the term frequency (TF) matrix for a list of documents.
    Each row corresponds to a document, and each column to a unique term across all documents.
    """
    # Collect vocabulary and term frequencies
    vocab = {}
    tf_counts = []
    for doc in documents:
        doc = clean_text(doc)
        words = doc.split()
        term_count = defaultdict(int)
        for word in words:
            if word not in vocab:
                vocab[word] = len(vocab)  # assign index if word is new
            term_count[word] += 1
        tf_counts.append(term_count)

    # Initialize a term-frequency matrix with zeros
    tf_matrix = np.zeros((len(documents), len(vocab)))
    for doc_index, term_count in enumerate(tf_counts):
        for term, count in term_count.items():
            tf_matrix[doc_index, vocab[term]] = count / len(documents[doc_index].split())

    return tf_matrix, vocab


def compute_idf_vector(documents: List[str], vocab: Dict[str, int]) -> np.ndarray:
    """
    Compute the inverse document frequency (IDF) vector for each term in the vocabulary.
    """
    num_docs = len(documents)
    idf_vector = np.zeros(len(vocab))
    doc_occurrences = defaultdict(int)

    for doc in documents:
        doc = clean_text(doc)
        unique_terms = set(doc.split())
        for term in unique_terms:
            if term in vocab:
                doc_occurrences[term] += 1

    for term, idx in vocab.items():
        idf_vector[idx] = math.log(num_docs / (1 + doc_occurrences[term]))  # Smoothed IDF

    return idf_vector


def compute_tf_idf_matrix(documents: List[str]) -> Tuple[np.ndarray, dict]:
    """
    Combine TF and IDF to create the TF-IDF matrix for the documents.
    """
    tf_matrix, vocab = compute_tf_matrix(documents)
    idf_vector = compute_idf_vector(documents, vocab)
    tf_idf_matrix = tf_matrix * idf_vector  # element-wise multiplication for TF-IDF scores

    return tf_idf_matrix, vocab


def rank_by_tf_idf(query: str, documents: List[str], n: int) -> List[str]:
    """
    Rank documents based on their cosine similarity to the query's TF-IDF vector.
    """
    # Clean the query text
    query = clean_text(query)

    # Generate TF-IDF matrix for the documents and vocabulary
    tf_idf_matrix, vocab = compute_tf_idf_matrix(documents)

    # Generate query vector based on the document vocabulary
    query_terms = query.split()
    query_tf_vector = np.zeros(len(vocab))
    for term in query_terms:
        if term in vocab:
            query_tf_vector[vocab[term]] += 1 / len(query_terms)  # normalize by query length

    # Recalculate query vector using the IDF of each term
    idf_vector = compute_idf_vector(documents, vocab)
    query_tf_idf_vector = query_tf_vector * idf_vector  # element-wise multiplication

    # Calculate cosine similarity between the query and each document
    similarities = []
    for doc_vector in tf_idf_matrix:
        cosine_similarity = np.dot(query_tf_idf_vector, doc_vector) / (
                np.linalg.norm(query_tf_idf_vector) * np.linalg.norm(doc_vector)
        )
        similarities.append(cosine_similarity)

    # Sort documents by similarity and return the top N
    ranked_indices = np.argsort(similarities)[::-1][:n]
    return [documents[i] for i in ranked_indices]


def get_bert_embedding(text: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_state = outputs.last_hidden_state

    # Take the mean of all token embeddings (not just the last token)
    # We are now averaging over all token embeddings to get a fixed-size representation for the document
    mean_embedding = last_hidden_state.mean(dim=1)

    return mean_embedding.squeeze(0)  # Squeeze to remove the batch dimension


def bert_re_rank(query: str, ranked_docs: List[str], top_k: int) -> List[str]:
    query_embedding = get_bert_embedding(query)
    doc_embeddings = [get_bert_embedding(doc) for doc in ranked_docs]

    # Calculate cosine similarity between query and document embeddings
    similarities = [(doc, torch.cosine_similarity(query_embedding, doc_embedding, dim=0).item())
                    for doc, doc_embedding in zip(ranked_docs, doc_embeddings)]

    top_docs = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    return [doc[0] for doc in top_docs]


def fetch_relevant_jobs(question: str) -> List[int]:
    documents = job_listings_strings
    top_docs = rank_by_tf_idf(question, documents, n=5)
    re_ranked_docs = bert_re_rank(question, top_docs, top_k=5)
    return [jobs.get_id_from_job_string(job) for job in re_ranked_docs]
