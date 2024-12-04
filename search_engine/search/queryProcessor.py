import numpy as np
from scipy import sparse
import pandas as pd
from sentence_transformers import SentenceTransformer

class QueryProcessor:
    def __init__(self, doc_vectors_path: str, model: str):
        self.doc_matrix = np.load(doc_vectors_path)
        self.model = SentenceTransformer(model)

    def compute_query_vector(self, query: str) -> np.array:
        query_vector =  self.model.encode(query, convert_to_numpy=True)
        return query_vector

    def compute_scores(self, query: str):
        query_vector = self.compute_query_vector(query)
        scores = np.matmul(self.doc_matrix, query_vector)
        return scores
    
    def process_query(self, query: str, k: int):
        scores = self.compute_scores(query)
        best = np.argsort(scores)[-k:][::-1]
        return [int(doc+1) for doc in best]
    
    def _process_queries(self, queries: list[str], k: int, similarity_func):
        query_embeddings = self.model.encode(queries, convert_to_numpy=True)
        scores_matrix = similarity_func(self.doc_matrix, query_embeddings)
        sorted = np.argsort(scores_matrix, axis=0)
        topk = sorted[-k:][::-1]
        result = topk + np.ones(topk.shape, topk.dtype)
        return result.T
    
    # Returns a matrix containing the document rankings for each query
    # A document ranking for a query is stored in a row
    # e.g. result[0] is the ranking for queries[0]
    def process_queries(self, queries: list[str], k: int):
        def similarity_func(document_embeddings, query_embeddings):
            query_embeddings = query_embeddings.T # put the query vectors in columns
            scores_matrix = np.matmul(document_embeddings, query_embeddings)
            return scores_matrix
        return self._process_queries(queries, k, similarity_func)
    
    # Same function as process_queries, but uses the model's built-in similarity function
    # instead of our own cosine implementation.
    # This function and process_queries should return exactly the same result. 
    def process_queries_using_lib_similarity(self, queries: list[str], k: int):
        def similarity_func(document_embeddings, query_embeddings):
            return self.model.similarity(query_embeddings, document_embeddings).numpy().T
        return self._process_queries(queries, k, similarity_func)
    
if __name__ == "__main__":
    queries = pd.read_csv("small_queries.csv")
    results = pd.read_csv("results_small.csv")
    queryProcessor = QueryProcessor("doc_vectors.npz", "all-MiniLM-L6-v2")
    for query_id, query in queries.itertuples(index=False):
        doc_ids = queryProcessor.process_query(query, 10)
        expected = results[results["Query_number"] == query_id]["doc_number"].values
        print(query, " found: ", doc_ids, end=" ")
        print("expected: ", expected)