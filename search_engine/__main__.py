"""
Information retrieval project 1

Authors: Jarne Besjes, Kobe De Broeck, Liam Leirs
"""
import os
import sys
import time
import argparse
import search_engine.preprocessing.documentVectorizer as documentVectorizer
from search_engine.search.queryProcessor import QueryProcessor
import numpy as np
import pandas as pd
from tqdm import tqdm


argparser = argparse.ArgumentParser(description='Information retrieval project 3')
argparser.add_argument('--query', type=str, help='The query to search for')
argparser.add_argument('-n', type=int, default=10, help='The number of results to return')
argparser.add_argument('--no-index', action='store_true', help='Do not index the documents')
argparser.add_argument('--docs-folder', type=str, help='The location of the folder where all documents are stored', required=True)
argparser.add_argument('--mode', type=str, help='Select mode: search or bench', required=True)

def calculate_precision_at_k(retrieved, relevant, k):
    count = 0
    sum = 0
    for i in range(1, k+1):
        if i <= len(retrieved) and retrieved[i - 1] in relevant:
            count += 1
            sum += count / i
    return sum / count if count > 0 else -1


def calculate_recall_at_k(retrieved, relevant, k):
    count = 0
    sum = 0
    for i in range(1, k+1):
        if i <= len(retrieved) and retrieved[i - 1] in relevant:
            count += 1
            sum += count / len(relevant)
    return sum / count if count > 0 else -1

if __name__ == "__main__":
    try:
        args = argparser.parse_args()
    except:
        argparser.print_help()
        sys.exit(1)

    query = args.query
    n = args.n
    docs_folder = args.docs_folder
    index_folder = docs_folder.rstrip("/") + "_index"
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
        # Check if the inverted index folder exists and is not older than 1 day
        if not os.path.exists("./" + index_folder) :
            print('Indexing...', file=sys.stderr)
            #indexer.run_indexer(docs_folder)
            print('Indexing done', file=sys.stderr)

    if bench:
        # benchmark
        queryProcessor = QueryProcessor("doc_vectors.npz", "all-MiniLM-L6-v2")
        queries = pd.read_csv("small_queries.csv")
        query_ids = queries["Query number"]

        avg_precisions = []
        avg_recalls = []

        expected_results = pd.read_csv("results_small.csv")

        for i, (query_id, query) in tqdm(enumerate(queries.itertuples(index=False)), total = len(queries), desc="Running Benchmark..."):
            for ki, k in enumerate([3, 10]):
                precisions = []
                recalls = []
                retrieved = queryProcessor.process_query(query, k)
                for i in range(k):
                    relevant = expected_results[expected_results["Query_number"] == query_id]["doc_number"].to_list()
                    precision = calculate_precision_at_k(retrieved[:i], relevant, k)
                    recall = calculate_recall_at_k(retrieved[:i], relevant, k)
                    if precision != -1:
                        precisions.append(precision)
                    if recall != -1:
                        recalls.append(recall)
                avg_precisions.append(np.mean(precisions))
                avg_recalls.append(np.mean(recalls))
            
        for i in range(2):
            print(f"Mean avg precision at {3 if i == 0 else 10}: ", np.mean(avg_precisions[i]))
            print(f"Mean avg recall at {3 if i == 0 else 10}: ", np.mean(avg_recalls[i]))

    else:
        # search
        queryProcessor = QueryProcessor(index_folder)
        results = queryProcessor.get_top_k_results(query, n)
        print(results)