import numpy as np
from sentence_transformers import SentenceTransformer
from scipy import sparse
import os
from tqdm import tqdm
import time
from sklearn.cluster import KMeans

def count_txt_files_in_folder(folder_path):
    with os.scandir(folder_path) as entries:
        txt_file_count = sum(1 for entry in entries if entry.is_file() and entry.name.endswith('.txt'))
    return txt_file_count

class DocumentVectorizer:
    def __init__(self, model: str, out_path: str):
        self.model = SentenceTransformer(model)
        self.out_path = out_path

    def compute_doc_matrix(self, directory: str) -> np.matrix:
        documents = []
        document_count = count_txt_files_in_folder(directory)
        for i in tqdm(range(1, document_count+1), desc="Reading documents..."):
            with open(f"{directory}/output_{i}.txt") as file:
                text = file.read()
                documents.append(text)
        
        print("Encoding documents...")
        encoding_start_time = time.time()
        document_embeddings = self.model.encode(documents, convert_to_numpy=True)
        print(f"Done encoding documents. Elapsed: {(time.time() - encoding_start_time):.2f} seconds")
        
        print("Clustering...")
        clustering_start_time = time.time()
        kmeans = KMeans(n_clusters=10, random_state=42)
        kmeans.fit(document_embeddings)
        cluster_assignments = kmeans.labels_
        centroids = kmeans.cluster_centers_
        print(f"Done clustering. Elapsed: {(time.time() - clustering_start_time):.2f} seconds")

        os.makedirs(self.out_path, exist_ok=True)
        
        np.save(f"{self.out_path}/centroids.npy", centroids)

        for ci in range(0, centroids.shape[0]):
            cluster_doc_ids = np.where(cluster_assignments == ci)
            cluster_vecs = document_embeddings[cluster_doc_ids]
            np.save(f"{self.out_path}/cluster_{ci}.npy", cluster_vecs)
            np.save(f"{self.out_path}/cluster_doc_ids_{ci}.npy", cluster_doc_ids)
        
        # np.save(f"{self.out_path}.npy", document_embeddings)

if __name__ == "__main__":
    vectorizer = DocumentVectorizer("all-MiniLM-L6-v2", "doc_vectors")
    vectorizer.compute_doc_matrix('full_docs_small')
