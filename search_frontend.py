import bs4
from flask import Flask, request, jsonify
import sys
import requests
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
from ranking_methods import *
import google


idx_body = InvertedIndex.read_index('/home/emilyt/ir_project_212270458_211425426/body', 'body_index')
idx_title = InvertedIndex.read_index('/home/emilyt/ir_project_212270458_211425426/title', 'title_index')
idx_anchor = InvertedIndex.read_index('/home/emilyt/ir_project_212270458_211425426/anchor', 'anchor_index')


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


# with open('table.csv', 'r') as f:
#     readit = csv.reader(f)
#     thedata = list(readit)
#     dict_data = dict([(int(i), float(j)) for i, j in thedata])
# with open('page_rank.json', 'w') as fp:
#     json.dump(dict_data, fp)
# with open('page_rank.json', 'r') as f:
#     page_rank = json.load(f)


@app.route("/search")
def search():
    """ Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    bm25_body = BM25_from_index(idx_body, '/home/emilyt/ir_project_212270458_211425426/body/')  # create BM25 object
    res = bm25_body.search(query)
    # END SOLUTION
    return jsonify(res)


@app.route("/search_body")
def search_body():
    """ Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = top_n_by_cosine_similarity(query, idx_body, '/home/emilyt/ir_project_212270458_211425426/body/', 100)
    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    """ Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        QUERY WORDS that appear in the title. For example, a document with a
        title that matches two of the query words will be ranked before a
        document with a title that matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = binary_search(query, idx_title, '/home/emilyt/ir_project_212270458_211425426/title/')
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    """ Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        For example, a document with a anchor text that matches two of the
        query words will be ranked before a document with anchor text that
        matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = binary_search(query, idx_anchor, '/home/emilyt/ir_project_212270458_211425426/anchor/')
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    """ Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    """
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    with open('/home/emilyt/ir_project_212270458_211425426/page_rank.json', 'r') as f:
        page_rank = json.load(f)
    for doc_id in wiki_ids:
        if str(doc_id) in page_rank:
            res.append(page_rank[str(doc_id)])
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    """ Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    """
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    page_view_pk = pd.read_pickle(r'/home/emilyt/ir_project_212270458_211425426/pageviews-202108-user.pkl')
    for doc_id in wiki_ids:
        if doc_id in page_view_pk:
            res.append(page_view_pk[doc_id])
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080)

