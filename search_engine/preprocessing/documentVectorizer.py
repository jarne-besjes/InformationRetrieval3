import numpy as np
from sentence_transformers import SentenceTransformer
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

    def split_into_windows(self, text, window_size=512, step=256):
        """Splits text into overlapping windows."""
        tokens = text.split()  # Use simple whitespace-based tokenization for demonstration
        windows = []
        for i in range(0, len(tokens), step):
            window = tokens[i:i + window_size]
            if window:  # Avoid adding empty windows
                windows.append(" ".join(window))
        return windows

    def generate_doc_embedding(self, text: str):
        chunks = self.split_into_windows(text)
        embeddings = self.model.encode(chunks)
        doc_embedding = np.mean(embeddings, axis=0)
        return doc_embedding

    def compute_doc_matrix(self, directory: str) -> np.matrix:
        sentences = []
        embeddings = []
        document_count = count_txt_files_in_folder(directory)
        for i in tqdm(range(1, document_count+1), desc="Reading documents..."):
            with open(f"{directory}/output_{i}.txt") as file:
                text = file.read()
                if len(text) == 0:
                    text = 'F' # Empty documents are problematic, hence this unfortunate hack, F in the chat...
                sentences.append(text)
        for sentence in sentences:
            embedding = self.generate_doc_embedding(sentence)
            embeddings.append(embedding)
        document_embeddings = np.array(embeddings)
        np.save(f"{self.out_path}.npy", document_embeddings)

if __name__ == "__main__":
    vectorizer = DocumentVectorizer("all-MiniLM-L6-v2", "doc_vectors")
    vectorizer.compute_doc_matrix('full_docs_small')
