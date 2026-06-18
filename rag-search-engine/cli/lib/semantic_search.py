import argparse
import json

from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import re as re

class SemanticSearch:
  def __init__(self):
    self.model = SentenceTransformer('all-MiniLM-L6-v2')
    self.embeddings = None
    self.documents = None
    self.document_map = {}
    self._base_dir = Path("lib") 
    self._cache_dir = self._base_dir.parent / "cache"
    self._data_path = self._base_dir.parent / "data"/ "movies.json"

  def generate_embedding(self, text: str):
    if text is None:
      raise ValueError("Input text cannot be None")
    embedding = self.model.encode([text])
    return embedding[0]

  def build_embeddings(self, documents: list):
    self.documents = documents
    doc_list = []
    for doc in documents:
      doc_id = doc.get("id")
      self.document_map[doc_id] = doc
      doc_list.append(f"{doc['title']}: {doc['description']}")
    
    self.embeddings = self.model.encode(doc_list, show_progress_bar=True)
    np.save(f"{self._cache_dir}/movie_embeddings.npy", self.embeddings)
    return self.embeddings

  def load_or_create_embeddings(self, documents: list):
    self.documents = documents
    for doc in documents:
      doc_id = doc.get("id")
      self.document_map[doc_id] = doc
    if (self._cache_dir / "movie_embeddings.npy").exists():
      self.embeddings = np.load(f"{self._cache_dir}/movie_embeddings.npy")
    else:
      self.embeddings = self.build_embeddings(documents)
    return self.embeddings

  def search(self, query: str, top_k: int = 5):
    documents = load_documents(self)
    embeddings  = self.load_or_create_embeddings(documents)

    if embeddings is None or documents is None:
      raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
    
    query_embedding = self.generate_embedding(query)

    # Compute similarities for each document by comparing the query vector to each embedding
    similarities = np.array([cosine_similarity(query_embedding, embeddings[i]) for i in range(len(documents))])

    # Ensure similarities is a 1-D numpy array
    similarities = np.asarray(similarities)
    if similarities.ndim != 1 or similarities.shape[0] != len(documents):
      raise ValueError("Unexpected similarity shape")

    top_k = min(top_k, len(documents))
    top_k_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for i in top_k_indices:
      doc = documents[i]
      score = float(similarities[i])
      results.append({
        "title": doc.get("title", ""),
        "description": doc.get("description", ""),
        "score": score
      })
    
    return results

def load_documents(semantic_search: SemanticSearch):
  with open(semantic_search._data_path, "r", encoding='utf-8') as file:
    documents = json.load(file)
  return documents.get("movies", [])
    
def verify_model():
  semantic_search = SemanticSearch()
  print(semantic_search.model)

def embed_text(text: str):
  semantic_search = SemanticSearch()
  embedding = semantic_search.generate_embedding(text)
  print(f"Text: {text}")
  print(f"First 3 dimensions: {embedding[:3]}")
  print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
  semantic_search = SemanticSearch()
  document = load_documents(semantic_search)
  embeddings = semantic_search.load_or_create_embeddings(document)
  print(f"Number of docs:   {len(document)}")
  print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def embed_query_text(query: str):
  semantic_search = SemanticSearch()
  embedding = semantic_search.generate_embedding(query)
  print(f"Query: {query}")
  print(f"First 3 dimensions: {embedding[:3]}")
  print(f"Dimensions: {embedding.shape[0]}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
  dot_product = np.dot(vec1, vec2)
  norm_vec1 = np.linalg.norm(vec1)
  norm_vec2 = np.linalg.norm(vec2)
  if norm_vec1 == 0 or norm_vec2 == 0:
    return 0.0
  return float(dot_product / (norm_vec1 * norm_vec2))

def search(query: str, top_k: int = 5):
  semantic_search = SemanticSearch()
  results = semantic_search.search(query, top_k=top_k)
  for index, result in enumerate(results):
    print(f"{index+1}. {result['title']} (score: {result['score']:.4f})")
    print(f"\n{result['description']}\n")

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 0):
  words = text.split()
  result = []
  start = 0
  print(f"Chunking {len(text)} characters")
  while start < len(words):
    chunk = " ".join(words[start:start + chunk_size]) # list[start:end]  start is included / end is excluded
    if len(chunk.split()) <= overlap:
      break
    result.append(chunk)
    start += chunk_size - overlap
  
  for i, chunk in enumerate(result, start=1):
    print(f"{i}. {chunk}")


def semantic_chunk(text: str, max_chunk_size: int = 4, overlap: int = 0):
  reg_ex = r"(?<=[.!?])\s+"
  text = text.strip()
  if not text:
    return []
  
  words = re.split(reg_ex, text) or []
  start = 0
  result = []

  print(f"Semantically chunking {len(text)} characters")

  while start < len(words):
    chunk = " ".join(words[start:start+max_chunk_size]).strip()
    if len(re.split(reg_ex, chunk)) <= overlap:
      break
    if chunk:
      result.append(chunk)
    start += max_chunk_size - overlap
  
  for i in range(len(result)):
    print(f"{i+1}. {result[i]}\n")
    
  return result
