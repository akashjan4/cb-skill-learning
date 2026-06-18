from .semantic_search import SemanticSearch, semantic_chunk, cosine_similarity
import numpy as np
import json


class ChunkedSemanticSearch(SemanticSearch):
  def __init__(self) -> None:
    super().__init__()
    self.chunk_embeddings = None
    self.chunk_metadata = None

  def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
    """
    Build embeddings for chunked document descriptions.
    - Chunk each description into 4-sentence chunks with 1-sentence overlap.
    - Generate embeddings for all chunks.
    - Save embeddings and metadata to cache.
    """
    self.documents = documents
    all_chunks = []
    chunk_metadata = []

    # Process each document
    for movie_idx, doc in enumerate(documents):
      doc_id = doc.get("id")
      desc = doc.get("description", "").strip()

      # Skip if description is empty
      if not desc:
        continue

      # Populate document map
      self.document_map[doc_id] = doc

      # Split description into semantic chunks (4 sentences, 1 sentence overlap)
      chunks = semantic_chunk(desc, max_chunk_size=4, overlap=1)

      # Add each chunk and its metadata
      for chunk_idx, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        chunk_metadata.append({
          "movie_idx": movie_idx,
          "chunk_idx": chunk_idx,
          "total_chunks": len(chunks),
          "doc_id": doc_id
        })

    # Generate embeddings for all chunks
    if all_chunks:
      self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
    else:
      self.chunk_embeddings = np.array([])

    # Save embeddings to cache
    np.save(f"{self._cache_dir}/chunk_embeddings.npy", self.chunk_embeddings)

    # Save metadata to cache
    with open(f"{self._cache_dir}/chunk_metadata.json", "w", encoding='utf-8') as file:
      json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, file, indent=2)

    self.chunk_metadata = chunk_metadata
    return self.chunk_embeddings

  def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
    """
    Load chunk embeddings from cache or create them if they don't exist.
    - Populate self.documents and self.document_map.
    - If cache files exist, load them; otherwise rebuild.
    """
    self.documents = documents

    # Populate document map from input documents
    for doc in documents:
      doc_id = doc.get("id")
      self.document_map[doc_id] = doc

    # Check if cache files exist
    chunk_embeddings_path = self._cache_dir / "chunk_embeddings.npy"
    chunk_metadata_path = self._cache_dir / "chunk_metadata.json"

    if chunk_embeddings_path.exists() and chunk_metadata_path.exists():
      # Load from cache
      self.chunk_embeddings = np.load(f"{self._cache_dir}/chunk_embeddings.npy")
      with open(f"{self._cache_dir}/chunk_metadata.json", "r", encoding='utf-8') as file:
        metadata_dict = json.load(file)
        self.chunk_metadata = metadata_dict.get("chunks", [])
      return self.chunk_embeddings
    else:
      # Build and cache
      return self.build_chunk_embeddings(documents)
  
  def search_chunks(self, query:str, limit:int = 10):
    if self.chunk_embeddings is None or len(self.chunk_embeddings) == 0:
      return []
    embedding  = self.generate_embedding(query)
    chunk_scores = []
    for i ,chunk_e in enumerate(self.chunk_embeddings):
      score = cosine_similarity(embedding, chunk_e)
      chunk_scores.append({
        "chunk_idx":self.chunk_metadata[i]['chunk_idx'],
        "movie_idx":self.chunk_metadata[i]['movie_idx'],
        "score":score
      })
      
    movie_scores = {}
    for chunk_score in chunk_scores:
      movie_idx = chunk_score['movie_idx']
      score = chunk_score['score']
      
      if movie_idx not in movie_scores or score > movie_scores[movie_idx]['score']:
        movie_scores[movie_idx] = chunk_score
    
    sorted_movies = sorted(movie_scores.values(), key=lambda x: x['score'], reverse=True)
    top_results = sorted_movies[:limit]
    
    results = []
    for movie_result in top_results:
      movie_idx = movie_result['movie_idx']
      doc = self.documents[movie_idx]
      results.append({
        "doc_id":doc.get('id'),
        "title":doc.get('title'),
        "document":doc.get('description', '')[:100],
        "score":round(movie_result['score'], 2),
        "metadata":doc.get('metadata', {})
      })
    return results

    

    