import numpy as np
from sentence_transformers import SentenceTransformer
from scipy import sparse
import os

class DocumentVectorizer:
    def __init__(self, model: str, out_path: str):
        self.model = SentenceTransformer(model)
        self.out_path = out_path
        
    def compute_doc_matrix(self, directory: str) -> np.matrix:
        doc_vectors = []
        for name in os.listdir(directory):
            if not name.endswith(".txt"):
                continue
            with open(os.path.join(directory, name)) as file:
                text = file.read()
                doc_vector = self.model.encode(text)
                doc_vectors.append(doc_vector)
        doc_vectors = np.stack(doc_vectors, axis=1)
        doc_vectors = sparse.csr_matrix(doc_vectors)
        sparse.save_npz(self.out_path, doc_vectors)

if __name__ == "__main__":
    vectorizer = DocumentVectorizer("all-MiniLM-L6-v2", "doc_vectors")
    vectorizer.compute_doc_matrix("full_docs_small")

