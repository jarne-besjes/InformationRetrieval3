import numpy as np
from sentence_transformers import SentenceTransformer
from scipy import sparse
import os
from tqdm import tqdm
import time
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
import shutil

def count_txt_files_in_folder(folder_path):
    with os.scandir(folder_path) as entries:
        txt_file_count = sum(1 for entry in entries if entry.is_file() and entry.name.endswith('.txt'))
    return txt_file_count

class DocumentVectorizer:
    def __init__(self, model: str, out_path: str):
        self.model = SentenceTransformer(model)
        self.out_path = out_path

    def compute_doc_matrix(self, directory: str, clustering: bool = False) -> np.matrix:
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

        os.makedirs(self.out_path, exist_ok=True)
        
        np.save(f"{self.out_path}/document_embeddings.npy", document_embeddings)

        if clustering:
            self.do_clustering(document_embeddings, 10)
    
    def do_clustering(self, document_embeddings, cluster_count: int):
        print("Clustering...")
        clustering_start_time = time.time()
        kmeans = KMeans(n_clusters=cluster_count, random_state=42)
        kmeans.fit(document_embeddings)
        cluster_assignments = kmeans.labels_
        centroids = kmeans.cluster_centers_ 
        centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        print(f"Done clustering. Elapsed: {(time.time() - clustering_start_time):.2f} seconds")

        cluster_data_path = f"{self.out_path}/cluster_data"

        # Create or overwrite cluster_data output folder
        if os.path.exists(cluster_data_path):
            shutil.rmtree(cluster_data_path)
        os.makedirs(cluster_data_path)

        np.save(f"{cluster_data_path}/centroids.npy", centroids)

        for ci in range(0, centroids.shape[0]):
            cluster_doc_ids = np.where(cluster_assignments == ci)
            cluster_vecs = document_embeddings[cluster_doc_ids]
            np.save(f"{cluster_data_path}/cluster_{ci}.npy", cluster_vecs)
            np.save(f"{cluster_data_path}/cluster_doc_ids_{ci}.npy", cluster_doc_ids)

import argparse
argparser = argparse.ArgumentParser(description='Document Vectorizer')
argparser.add_argument('--mode', type=str, help='Select mode: cluster(=cluster existing embeddings file)', required=True)
argparser.add_argument('--index-folder', type=str, help='The location of the folder where document embeddings file is stored', required=True)
argparser.add_argument('--cluster-count', type=int, default=10, required=False, help='Create cluster-count clusters for the *existing* document_embeddings.npy file stored in index-folder')

if __name__ == "__main__":
    try:
        args = argparser.parse_args()
    except:
        argparser.print_help()
        sys.exit(1)

    index_folder = args.index_folder
    mode = args.mode

    if mode == "cluster":
        print("Cluster mode selected")
        vectorizer = DocumentVectorizer("all-MiniLM-L6-v2", index_folder)
        try:
            document_embeddings = np.load(f"{index_folder}/document_embeddings.npy")
        except FileNotFoundError:
            print(f"Error: The file '{index_folder}/document_embeddings.npy' does not exist. Please check the path and try again.")
            sys.exit(1)
        except Exception as e:
            print(f"An error occurred reading the document embeddings npy file: {e}")
            sys.exit(1)
        cluster_count = args.cluster_count
        if cluster_count is None:
            print('Please provide a cluster count', file=sys.stderr)
            sys.exit(1)
        vectorizer.do_clustering(document_embeddings, cluster_count)
    else:
        print('Please provide a valid mode', file=sys.stderr)
        sys.exit(1)
