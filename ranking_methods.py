from flask import Flask, request, jsonify
import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby, chain
import pandas as pd
import numpy as np
import os
import re
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from time import time
from timeit import timeit
from pathlib import Path
import pickle
from google.cloud import storage
from contextlib import closing
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict, Counter
from tqdm import tqdm
import operator
import json
import csv
from io import StringIO
import math
import hashlib
from collections import Counter, OrderedDict, defaultdict
import heapq
from inverted_index_gcp import *
# from inverted_index_colab import *
from search_frontend import *

nltk.download('stopwords')

with open('/home/emilyt/ir_project_212270458_211425426/doc_to_title.json', 'r') as f:
    titles = json.load(f)
with open('/home/emilyt/ir_project_212270458_211425426/page_rank.json', 'r') as f:
    page_rank = json.load(f)


def tokenize(text):
    """
    Get text as string and return list of tokens without stopwords
    """
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ["category", "references", "also", "external", "links",
                        "may", "first", "see", "history", "people", "one", "two",
                        "part", "thumb", "including", "second", "following",
                        "many", "however", "would", "became"]

    all_stopwords = english_stopwords.union(corpus_stopwords)
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
    return list_of_tokens


def binary_search(query_to_search, index, f_name):
    """
    Returns all documents that has at least one of the query's tokens,
    sorted when the first document is that who has the highest number of query's tokens
    """
    result_dict = {}
    tokens = np.unique(tokenize(query_to_search))  # unique tokens in the query
    for token in tokens:
        if token in index.df:  # if the token is in the corpus
            docs_and_pls = index.read_posting_list(token, f_name)
            for doc_id, pls in docs_and_pls:
                result_dict[doc_id] = result_dict.get(doc_id, 0) + 1
    # sort result by the number of tokens appearance
    res = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
    return [(int(doc_id), titles[str(doc_id)]) for doc_id, score in res]


def generate_query_tfidf_vector(query_to_search, index):
    """
    Generate a vector representing the query
    """
    epsilon = .0000001
    Q = np.zeros(len(query_to_search))  # generate vector with the length of the query
    counter = Counter(query_to_search)
    ind = 0
    for token in np.unique(query_to_search):
        if token in list(index.term_total.keys()):  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divided by the length of the query
            idf = math.log((len(index.DL)) / (index.df[token] + epsilon), 10)  # smoothing
            Q[ind] = tf * idf
            ind += 1
    return Q


def get_candidate_documents_and_scores(query_to_search, index, f_name):
    """
    Generate a dictionary representing a pool of candidate documents for a given query.
    """
    candidates = {}
    N = len(index.DL)  # corpus size
    # iterate over all query terms and calculate their tf-idf values per document
    for term in np.unique(query_to_search):
        try:
            pls = index.read_posting_list(term, f_name)
        except:
            continue
        normalized_tfidf = []
        for doc_id, freq in pls:
            try:
                normalized_tfidf.append((doc_id, (freq / index.DL[doc_id]) * math.log(N / index.df[term], 10)))
            except:
                normalized_tfidf.append((doc_id, 0.0))
        for doc_id, tfidf in normalized_tfidf:
            if (doc_id, term) in candidates:
                candidates[(doc_id, term)] += tfidf
            else:
                candidates[(doc_id, term)] = tfidf

    return candidates


def cosine_similarity(doc_scores, Q, dl):
    """
    Calculate the cosine similarity value for certain document (doc_id) and query.
    doc_scores is a list of tf-idf scores for each term in the document.
    """
    scores = np.array(doc_scores)
    Q = np.array(Q)
    upper = np.dot(scores, Q)
    lower = len(Q) * dl
    if lower == 0.0:
        cos_sim_value = 0.0
    else:
        cos_sim_value = upper / lower
    return cos_sim_value


def top_n_by_cosine_similarity(query_to_search, index, f_name, N=10):
    """
    returns:
      list of top N documents sorted by score as (doc_id, title)
    """
    tokens = tokenize(query_to_search)
    # all documents that contains query term and their tf-idf scores
    candidates_scores = get_candidate_documents_and_scores(tokens, index, f_name)
    # all unique relevant documents
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])

    # {doc_id: ["score1", "score2"]}
    scores_per_doc = dict((doc_id, [0] * len(tokens)) for doc_id in unique_candidates)
    token_idx = dict.fromkeys(token for token in np.unique(tokens))
    token_idx.update((k, i) for i, k in enumerate(token_idx))  # {term1: 0, term2: 1, term3: 2, ...}
    for key, tfidf in candidates_scores.items():  # key = (doc_id, term)
        doc_id, term = key
        if doc_id == 0:
            continue
        tok_ind = token_idx[term]  # index of token where to insert the term score
        if doc_id in scores_per_doc:
            scores_per_doc[doc_id][tok_ind] = tfidf
        else:
            scores_per_doc[doc_id] = [0] * len(tokens)
            scores_per_doc[doc_id][tok_ind] = tfidf

    Q = generate_query_tfidf_vector(tokens, index)
    # list that we would refer to it as heap queue. In that way, instead of saving all documents and scores and
    # then sort them, we will keep just the best N documents and each time replace the worst by better document
    top_n_heap = []
    for doc_id, doc_scores in scores_per_doc.items():
        if doc_id == 0:
            continue
        cos_sim_val = cosine_similarity(doc_scores, Q, index.DL[doc_id])
        if len(top_n_heap) == N:
            heapq.heapify(top_n_heap)
            min_val = heapq.heappop(top_n_heap)  # (cos_sim, doc_id)
            if cos_sim_val > min_val[0]:
                heapq.heappush(top_n_heap, (cos_sim_val, doc_id))
            else:
                heapq.heappush(top_n_heap, min_val)
                top_n_heap = list(top_n_heap)
        else:
            heapq.heapify(top_n_heap)
            heapq.heappush(top_n_heap, (cos_sim_val, doc_id))
            top_n_heap = list(top_n_heap)

    return sorted([(int(doc_id), titles[str(doc_id)]) for score, doc_id in top_n_heap], key=lambda x: x[1], reverse=True)


def get_candidate_documents(query_to_search, index, f_name):
    candidates = []
    for term in np.unique(query_to_search):
        try:
            pls = index.read_posting_list(term, f_name)  # get posting list of the term if exists
        except:
            continue
        candidates += pls
    res = defaultdict(int)
    for cand, freq in candidates:
        res[cand] += int(freq)
    # we will return only candidates that the frequency of the query's terms there is
    # higher then some threshold we decided.
    return np.unique([cand for cand, freq in res.items() if freq > (50 * len(query_to_search))])


class BM25_from_index:

    def __init__(self, index, f_name, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.f_name = f_name
        self.N = len(self.index.DL)
        self.AVGDL = sum(self.index.DL.values()) / self.N

    def calc_idf(self, list_of_tokens):
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def search(self, query, N=100):
        query_to_search = tokenize(query)
        self.idf = self.calc_idf(query_to_search)
        term_frequencies = {}  # {term1: {doc1: 4, doc2: 12, ...}, ...}
        for token in query_to_search:
            if token in self.index.df:
                try:
                    # save posting list as dictionary
                    term_frequencies[token] = dict(self.index.read_posting_list(token, self.f_name))
                except:
                    continue
        candidates = get_candidate_documents(query_to_search, self.index, self.f_name)
        scores = sorted([(doc_id, self._score(query_to_search, doc_id, term_frequencies)) for doc_id in candidates],
                        key=lambda x: x[1], reverse=True)[:N]
        doc_titles = [(int(doc_id), titles[str(doc_id)]) for doc_id, score in scores]
        return doc_titles

    def _score(self, query, doc_id, term_frequencies):
        score = 0.0
        if doc_id == 0:
            return score
        doc_len = self.index.DL[doc_id]
        for term in query:
            if doc_id in term_frequencies[term]:
                # calculate BM25 value
                freq = term_frequencies[term][doc_id]
                numerator = self.idf[term] * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + (self.b * doc_len / self.AVGDL))
                try:
                    score += (numerator / denominator) + math.log(page_rank[str(doc_id)])
                except:
                    score += (numerator / denominator)
        return score
