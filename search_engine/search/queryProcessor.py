import numpy as np
from scipy import sparse
import pandas as pd
from sentence_transformers import SentenceTransformer

class QueryProcessor:
    def __init__(self, doc_vectors_path: str, model: str):
        doc_matrix = sparse.load_npz(doc_vectors_path).toarray()
        norms = np.linalg.norm(doc_matrix, axis=0, keepdims=True)
        self.doc_matrix = doc_matrix / norms
        self.model = SentenceTransformer(model)

    def compute_query_vector(self, query: str) -> np.array:
        query_vector =  np.array(self.model.encode(query))
        norm = np.linalg.norm(query_vector, ord=2)
        if norm > 0:
            query_vector = query_vector / norm
        return query_vector

    def compute_scores(self, query: str):
        query_vector = self.compute_query_vector(query)
        scores = np.dot(query_vector, self.doc_matrix)
        return scores
    
    def process_query(self, query: str, k: int):
        scores = self.compute_scores(query)
        best = np.argsort(scores)[-k:][::-1]
        return [int(doc+1) for doc in best]
    
if __name__ == "__main__":
    queries = pd.read_csv("small_queries.csv")
    results = pd.read_csv("results_small.csv")
    queryProcessor = QueryProcessor("doc_vectors.npz", "all-MiniLM-L6-v2")
    for query_id, query in queries.itertuples(index=False):
        doc_ids = queryProcessor.process_query(query, 10)
        expected = results[results["Query_number"] == query_id]["doc_number"].values
        print(query, " found: ", doc_ids, end=" ")
        print("expected: ", expected)