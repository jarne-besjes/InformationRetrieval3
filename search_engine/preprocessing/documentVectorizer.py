import numpy as np
from sentence_transformers import SentenceTransformer
from scipy import sparse
import os
from tqdm import tqdm

def count_txt_files_in_folder(folder_path):
    with os.scandir(folder_path) as entries:
        txt_file_count = sum(1 for entry in entries if entry.is_file() and entry.name.endswith('.txt'))
    return txt_file_count

class DocumentVectorizer:
    def __init__(self, model: str, out_path: str):
        self.model = SentenceTransformer(model)
        self.out_path = out_path
        
    def compute_doc_matrix(self, directory: str) -> np.matrix:
        doc_vectors = []
        document_count = count_txt_files_in_folder(directory)
        for i in tqdm(range(1, document_count+1), desc="Reading documents..."):
            with open(f"{directory}/output_{i}.txt") as file:
                text = file.read()
                doc_vector = self.model.encode(text)
                doc_vectors.append(doc_vector)
        doc_vectors = np.stack(doc_vectors, axis=1)
        doc_vectors = sparse.csr_matrix(doc_vectors)
        sparse.save_npz(self.out_path, doc_vectors)

if __name__ == "__main__":
    vectorizer = DocumentVectorizer("all-MiniLM-L6-v2", "doc_vectors")
    vectorizer.compute_doc_matrix("full_docs_small")
