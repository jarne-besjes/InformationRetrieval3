import numpy as np
from sentence_transformers import SentenceTransformer
from scipy import sparse
import os
from tqdm import tqdm
import time

def count_txt_files_in_folder(folder_path):
    with os.scandir(folder_path) as entries:
        txt_file_count = sum(1 for entry in entries if entry.is_file() and entry.name.endswith('.txt'))
    return txt_file_count

class DocumentVectorizer:
    def __init__(self, model: str, out_path: str):
        self.model = SentenceTransformer(model)
        self.out_path = out_path

    def compute_doc_matrix(self, directory: str) -> np.matrix:
        document_parts = []
        document_count = count_txt_files_in_folder(directory)
        document_parts_counts = [] # tracks how many parts each document is split into
        for i in tqdm(range(1, document_count+1), desc="Reading documents..."):
            with open(f"{directory}/output_{i}.txt") as file:
                text = file.read()
                if len(text) == 0:
                    text = 'F' # Empty documents are problematic, hence this unfortunate hack, F in the chat...
                parts = self.split_into_parts(text)
                document_parts += parts
                document_parts_counts.append(len(parts))
        
        print(f"Number of documents that have been split: {sum([1 for count in document_parts_counts if count > 1])}")
        print(f"Encoding {len(document_parts)} document parts...")
        encoding_start_time = time.time()
        document_embeddings = self.model.encode(document_parts, convert_to_numpy=True)
        print(f"Done encoding documents. Elapsed: {(time.time() - encoding_start_time):.2f} seconds")
        
        # aggregation matrix will average the parts belonging to a document resulting in a single document vector
        # e.g. a row [0, 1/3, 1/3, 1/3, 0, ..., 0] will average the 3 parts embeddings of the second document
        aggregation_matrix = np.zeros(shape=(document_count, document_embeddings.shape[0]), dtype=document_embeddings.dtype)
        cur_doc_start = 0
        for i in range(0, document_count):
            document_parts_count = document_parts_counts[i]
            w = 1/document_parts_count
            # w = 1
            for j in range(0, document_parts_count):
                aggregation_matrix[i][cur_doc_start+j] = w
            cur_doc_start += document_parts_count

        document_embeddings = np.matmul(aggregation_matrix, document_embeddings)

        np.save(f"{self.out_path}.npy", document_embeddings)
    
    def split_into_parts(self, text: str) -> list[str]:
        words = text.split()
        parts = []
        words_per_part = 200
        part_start = 0
        while part_start < len(words):
            part_end = part_start + words_per_part
            if part_end > len(words):
                part_end = len(words)
            parts.append(" ".join(words[part_start:part_end]))
            part_start = part_end
        return parts

if __name__ == "__main__":
    vectorizer = DocumentVectorizer("all-MiniLM-L6-v2", "doc_vectors")
    vectorizer.compute_doc_matrix('full_docs_small')
