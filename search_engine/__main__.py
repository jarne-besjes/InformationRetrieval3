"""
Information retrieval project 1

Authors: Jarne Besjes, Kobe De Broeck, Liam Leirs
"""
import os
import sys
import time
import argparse

from sympy import Q
import search_engine.preprocessing.documentVectorizer as documentVectorizer
from search_engine.search.queryProcessor import QueryProcessor, QueryProcessorClustering
import numpy as np
import pandas as pd
from tqdm import tqdm
from search_engine import vars

k_map = {0: 1, 1:3, 2:5, 3: 10}


argparser = argparse.ArgumentParser(description='Information retrieval project 3')
argparser.add_argument('--query', type=str, help='The query to search for')
argparser.add_argument('-n', type=int, default=10, help='The number of results to return')
argparser.add_argument('--no-index', action='store_true', help='Do not index the documents')
argparser.add_argument('--reindex', action='store_true', help='Overwrite the existing index')
argparser.add_argument('--docs-folder', type=str, help='The location of the folder where all documents are stored', required=True)
argparser.add_argument('--mode', type=str, help='Select mode: search or bench', required=True)
argparser.add_argument('--clustering', action='store_true', help='Use clustering (for preprocessing & querying)')

def precision_at_k(retrieved, relevant, k):
    count = 0
    for i in range(1, k+1):
        if i <= len(retrieved) and retrieved[i - 1] in relevant:
            count += 1
    return count / k if count > 0 else 0

def recall_at_k(retrieved, relevant, k):
    count = 0
    for i in range(1, k+1):
        if i <= len(retrieved) and retrieved[i - 1] in relevant:
            count += 1
    return count / len(relevant) if count > 0 else 0

def average_precision_at_k(retrieved, relevant, k):
    count = 0
    sum = 0
    for i in range(1, k+1):
        if i <= len(retrieved) and retrieved[i - 1] in relevant:
            count += 1
            sum += count / i
    return sum / count if count > 0 else 0


def average_recall_at_k(retrieved, relevant, k):
    count = 0
    sum = 0
    for i in range(1, k+1):
        if i <= len(retrieved) and retrieved[i - 1] in relevant:
            count += 1
            sum += count / len(relevant)
    return sum / count if count > 0 else 0

if __name__ == "__main__":
    try:
        args = argparser.parse_args()
    except:
        argparser.print_help()
        sys.exit(1)

    query = args.query
    n = args.n
    docs_folder = args.docs_folder
    index_folder = docs_folder.rstrip("/") + "_embeddings"
    mode = args.mode

    if mode == "bench":
        bench = True
    elif mode == "search":
        bench = False
        if query is None:
            print('Please provide a query', file=sys.stderr)
            sys.exit(1)
    else:
        print('Please provide a valid mode', file=sys.stderr)
        sys.exit(1)

    if not args.no_index:
        output_path = index_folder
        
        indexing_needed =   args.reindex or \
                            (not os.path.exists(output_path)) or \
                            (args.clustering and not os.path.exists(f"{output_path}/cluster_data"))

        if indexing_needed:
            print('Preprocessing documents...', file=sys.stderr)
            vectorizer = documentVectorizer.DocumentVectorizer("all-MiniLM-L6-v2", output_path)
            vectorizer.compute_doc_matrix(docs_folder, clustering=args.clustering)
            print('Preprocessing done', file=sys.stderr)

    if bench:
        if args.clustering:
            queryProcessor = QueryProcessorClustering(index_folder, "all-MiniLM-L6-v2", clusters_to_evaluate=vars.CLUSTERS_TO_EVALUATE)
            # queryProcessor = QueryProcessorClustering("doc_vectors_BIG", "all-MiniLM-L6-v2", clusters_to_evaluate=3)
            queries_csv = pd.read_csv("dev_queries.tsv", sep='\t')
            expected_results = pd.read_csv("dev_query_results.csv")
            query_limit = vars.QUERY_LIMIT # The assigment says we may cut off at 1000 queries for Part 2
        else:
            queryProcessor = QueryProcessor(index_folder, "all-MiniLM-L6-v2")
            queries_csv = pd.read_csv("dev_small_queries.csv")
            expected_results = pd.read_csv("dev_query_results_small.csv")
            query_limit = None # no query limit for Part 1

        queries : list[tuple[int,str]] = []
        for (query_id, query) in queries_csv.itertuples(index=False):
            queries.append((query_id, query))
        # NOTE: sort the queries based on query_id to ensure deterministic query order across multiple runs
        queries.sort(key=lambda p: p[0])
        queries = queries[:query_limit]

        print("Scoring the documents...")
        queryProcessor.process_queries([query for (query_id, query) in queries], k=10)

        avg_precisions = [[], [], [], []]
        avg_recalls = [[], [], [], []]
        precisions = [[], [], [], []]
        recalls = [[], [], [], []]

        for query_i in tqdm(range(0, len(queries)), desc="Calculating precisions and recalls..."):
            query_id = queries[query_i][0]
            query_str = queries[query_i][1]
            retrieved_maxk = queryProcessor.get_query_top_docs(query_i)
            expected = expected_results[expected_results["Query_number"] == query_id]["doc_number"].values
            for i in range(4):
                precisions[i].append(precision_at_k(retrieved_maxk, expected, k_map[i]))
                recalls[i].append(recall_at_k(retrieved_maxk, expected, k_map[i]))
                avg_precisions[i].append(average_precision_at_k(retrieved_maxk, expected, k_map[i]))
                avg_recalls[i].append(average_recall_at_k(retrieved_maxk, expected, k_map[i]))

        for i in range(4):
            print(f"Avg precision at {k_map[i]}: ", np.mean(precisions[i]))
            print(f"Avg recall at {k_map[i]}: ", np.mean(recalls[i]))
        print("----------------------------")
        for i in range(4):
            print(f"Mean Avg precision at {k_map[i]}: ", np.mean(avg_precisions[i]))
            print(f"Mean Avg recall at {k_map[i]}: ", np.mean(avg_recalls[i]))
    else:
        # search
        queryProcessor = QueryProcessor(index_folder)
        results = queryProcessor.get_top_k_results(query, n)
        print(results)
