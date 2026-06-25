from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np
import json

class MultimodalSearch:
  def __init__(self, documents, model_name="clip-ViT-B-32"):
    self.model = SentenceTransformer(model_name)
    self.documents = documents
    self.texts = []
    for doc in documents:
      self.texts.append(f"{doc['title']}: {doc['description']}") 
      
    self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
  
  def __cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
      return 0.0
    return float(dot_product / (norm_vec1 * norm_vec2))
  
  def search_with_image(self, image_path):
    img_embedding = self.embed_image(image_path)
    
    results = []
    for i, text_embedding in enumerate(self.text_embeddings):
      similarity = self.__cosine_similarity(text_embedding, img_embedding)
      doc = self.documents[i]
      results.append({
        "id": doc.get("id"),
        "title": doc.get("title"),
        "description": doc.get("description"),
        "similarity": similarity
      })
      
    # Sort the results by similarity score, in descending order
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:5]
   
  def embed_image(self, image_path = "data/paddington.jpeg"):
    image = Image.open(image_path, "r")
    embedding = self.model.encode(image)
    return embedding


def verify_image_embedding(image_path:str):
  mms = MultimodalSearch([])
  embedding = mms.embed_image(image_path)
  print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path:str):
  with open("data/movies.json", "r", encoding='utf-8') as file:
    documents = json.load(file)
    documents_list = documents.get("movies", [])
    
  mms = MultimodalSearch(documents_list)
  return mms.search_with_image(image_path)
