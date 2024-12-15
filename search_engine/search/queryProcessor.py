import numpy as np
from scipy import sparse
import pandas as pd
from sentence_transformers import SentenceTransformer

class QueryProcessor:
    def __init__(self, doc_vectors_folder_path: str, model: str):
        self.doc_matrix = np.load(f"{doc_vectors_folder_path}/document_embeddings.npy")
        self.model = SentenceTransformer(model)
        self.document_rankings = None

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
    
    def process_queries(self, queries: list[str], k: int):
        self.document_rankings = self._process_queries_using_own_similarity(queries, k=k)
    
    def get_query_top_docs(self, query_index: int):
        if self.document_rankings is None:
            raise RuntimeError("get_query_top_docs called before process_queries. Provide the query list with process_queries first.")
        return self.document_rankings[query_index].tolist()

    # private methods
    # ------------------

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
    def _process_queries_using_own_similarity(self, queries: list[str], k: int):
        def similarity_func(document_embeddings, query_embeddings):
            query_embeddings = query_embeddings.T # put the query vectors in columns
            scores_matrix = np.matmul(document_embeddings, query_embeddings)
            return scores_matrix
        return self._process_queries(queries, k, similarity_func)
    
    # Same function as process_queries, but uses the model's built-in similarity function
    # instead of our own cosine implementation.
    # This function and process_queries should return exactly the same result. 
    def _process_queries_using_lib_similarity(self, queries: list[str], k: int):
        def similarity_func(document_embeddings, query_embeddings):
            return self.model.similarity(query_embeddings, document_embeddings).numpy().T
        return self._process_queries(queries, k, similarity_func)

class QueryProcessorClustering:
    def __init__(self, index_path: str, model: str, clusters_to_evaluate: int = 3):
        self.centroid_matrix = np.load(f"{index_path}/cluster_data/centroids.npy")
        self.model = SentenceTransformer(model)
        self.queries = None # filled in with process_queries
        self.k = None # filled in with process_queries
        self.t = clusters_to_evaluate
        cluster_count = self.centroid_matrix.shape[0]
        self.clusters = []
        self.cluster_doc_ids = []
        for ci in range(0, cluster_count):
            self.clusters.append(np.load(f"{index_path}/cluster_data/cluster_{ci}.npy"))
            self.cluster_doc_ids.append(np.load(f"{index_path}/cluster_data/cluster_doc_ids_{ci}.npy"))
    
    def _process_query(self, query_index: int, k: int):
        query_embedding = self.query_embeddings[query_index]
        topt_centroids = self.top_centroids_matrix[query_index]
        candidate_document_vectors = np.vstack([self.clusters[ci] for ci in topt_centroids])
        scores_matrix = np.matmul(candidate_document_vectors, query_embedding.T)
        doc_scores_sorted = np.argsort(scores_matrix, axis=0)
        topk_documents_local = doc_scores_sorted[-k:][::-1]
        # NOTE: map the "cluster-local" document indexes to the "global" document indexes
        doc_ids = np.vstack([self.cluster_doc_ids[ci].T for ci in topt_centroids]).ravel()
        topk_documents = doc_ids[topk_documents_local]
        result = topk_documents + np.ones(topk_documents.shape, topk_documents.dtype)
        return result
    
    def process_queries(self, queries: list[str], k: int):
        self.queries = queries
        self.k = k

        self.query_embeddings = self.model.encode(queries, convert_to_numpy=True)
        centroid_scores_for_all_queries = np.matmul(self.centroid_matrix, self.query_embeddings.T)
        centroid_scores_sorted = np.argsort(centroid_scores_for_all_queries, axis=0)
        # t is the number of candidate clusters
        self.top_centroids_matrix = centroid_scores_sorted[-self.t:][::-1].T
    
    def get_query_top_docs(self, query_index: int):
        if self.queries is None or self.k is None:
            raise RuntimeError("get_query_top_docs called before process_queries. Provide the query list with process_queries first.")
        return self._process_query(query_index, self.k)

if __name__ == "__main__":
    queries = pd.read_csv("small_queries.csv")
    results = pd.read_csv("results_small.csv")
    queryProcessor = QueryProcessor("doc_vectors.npz", "all-MiniLM-L6-v2")
    for query_id, query in queries.itertuples(index=False):
        doc_ids = queryProcessor.process_query(query, 10)
        expected = results[results["Query_number"] == query_id]["doc_number"].values
        print(query, " found: ", doc_ids, end=" ")
        print("expected: ", expected)