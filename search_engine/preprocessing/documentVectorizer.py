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

    def split_docs(self, text: str):
        max_length = self.model.tokenizer.model_max_length
        parts = [text[i: i+max_length] for i in range(0, len(text), max_length)]
        tokens = [token for part in parts for token in self.model.tokenizer.encode(part, truncation=False, add_special_tokens=False)]
        chunks = [tokens[i: i+max_length] for i in range(0, len(tokens), max_length)]
        return [self.model.tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

    def generate_doc_embedding(self, text: str):
        chunks = self.split_docs(text)
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
