import json
import os
import time
from dotenv import load_dotenv
from google import genai
from inverted_index import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch
from sentence_transformers import CrossEncoder
class HybridSearch:
  def __init__(self, documents: list[dict]) -> None:
    self.documents = documents
    self.semantic_search = ChunkedSemanticSearch()
    self.semantic_search.load_or_create_chunk_embeddings(documents)
    self.idx = InvertedIndex()
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
      raise RuntimeError("GEMINI_API_KEY environment variable not set")

    self.gemma_client = genai.Client(api_key=api_key)
    
    if not os.path.exists(self.idx._index_path):
      self.idx.build()
      self.idx.save()

  @staticmethod
  def normalize(scores: list) -> list[float]:
    if not scores:
      return []
    mn = min(scores)
    mx = max(scores)
    if mn == mx:
      return [1.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]

  @staticmethod
  def hybrid_score(keyword_score: float, semantic_score: float, alpha: float) -> float:
    return alpha * semantic_score + (1 - alpha) * keyword_score

  def _bm25_search(self, query: str, limit: int) -> list[dict]:
    return self.idx.bm25_search(query, limit)

  def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
    internal_limit = limit * 500

    bm25_results = self._bm25_search(query, internal_limit)
    bm25_scores = {doc_id: score for doc_id, score in bm25_results}

    semantic_results = self.semantic_search.search_chunks(query, internal_limit)
    semantic_scores = {}
    for r in semantic_results:
      doc_id = r["doc_id"]
      if isinstance(doc_id, str) and doc_id.isdigit():
        doc_id = int(doc_id)
      semantic_scores[doc_id] = r["score"]

    norm_bm25_list = self.normalize(list(bm25_scores.values()))
    norm_sem_list = self.normalize(list(semantic_scores.values()))

    doc_bm25 = dict(zip(bm25_scores.keys(), norm_bm25_list))
    doc_sem = dict(zip(semantic_scores.keys(), norm_sem_list))

    all_doc_ids = set(doc_bm25.keys()) | set(doc_sem.keys())

    doc_info = {}
    for doc_id in all_doc_ids:
      doc = self.idx.docmap.get(doc_id, {})
      kw_score = doc_bm25.get(doc_id, 0.0)
      sem_score = doc_sem.get(doc_id, 0.0)
      hybrid = self.hybrid_score(kw_score, sem_score, alpha)
      doc_info[doc_id] = {
        "doc": doc,
        "keyword_score": kw_score,
        "semantic_score": sem_score,
        "hybrid_score": hybrid,
      }

    sorted_docs = sorted(doc_info.values(), key=lambda x: x["hybrid_score"], reverse=True)

    results = []
    for entry in sorted_docs[:limit]:
      doc = entry["doc"]
      desc = doc.get("description", "")
      if len(desc) > 100:
        desc = desc[:100] + "..."
      results.append({
        "title": doc.get("title", "Unknown"),
        "description": desc,
        "hybrid_score": round(entry["hybrid_score"], 3),
        "bm25_score": round(entry["keyword_score"], 3),
        "semantic_score": round(entry["semantic_score"], 3),
      })
    return results

  def get_gemini_response(self, prompt):
    return self.gemma_client.models.generate_content(
      model="gemma-4-31b-it",
      contents=prompt
    )
  
  def gemini_llm(self, query, enhance):
    if not enhance:
      return 
    
    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
      Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
      Preserve punctuation and capitalization unless a change is required for a typo fix.
      If there are no spelling errors, or if you're unsure, output the original query unchanged.
      Output only the final query text, nothing else.
      User query: "{query}"
    """
    if enhance == "expand":
      prompt = f"""Expand the user-provided movie search query below with related terms.
        Add synonyms and related concepts that might appear in movie descriptions.
        Keep expansions relevant and focused.
        Output only the additional terms; they will be appended to the original query.

        Examples:
        - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
        - "action movie with bear" -> "action thriller bear chase fight adventure"
        - "comedy with bear" -> "comedy funny bear humor lighthearted"

        User query: "{query}"
        """

    response = self.get_gemini_response(prompt)
    
    print(f"Enhanced query ({enhance}): '{query}' -> '{response.text}'\n")
  
  def llm_base_reranking(self, query, doc):
    prompt = f"""Rate how well this movie matches the search query.
      Query: "{query}"
      Movie: {doc.get("title", "")} - {doc.get("document", "")}

      Consider:
      - Direct relevance to query
      - User intent (what they're looking for)
      - Content appropriateness

      Rate 0-10 (10 = perfect match).
      Output ONLY the number in your response, no other text or explanation.

      Score:"""

    return self.get_gemini_response(prompt)

  def llm_based_evaluation(self, query, formatted_results):
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:
      Query: "{query}"

      Results:
      {chr(10).join(formatted_results)}

      Scale:
      - 3: Highly relevant
      - 2: Relevant
      - 1: Marginally relevant
      - 0: Not relevant

      Do NOT give any numbers other than 0, 1, 2, or 3.

      Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

      [2, 0, 3, 2, 0, 1]"""

    return self.get_gemini_response(prompt)
 
  @staticmethod
  def rrf_score(rank: int, k: int) -> float:
    return 1.0 / (k + rank)

  def rrf_search(self, query: str, enhance: str, rerank_method: str, evaluate:bool, k: int, limit: int = 5) -> list[dict]:
    internal_limit = limit * 500
    if rerank_method:
      limit *= 5

    bm25_results = self._bm25_search(query, internal_limit)
    bm25_rank = {}
    print(f"query: {query}")
    
    for idx, (doc_id, _) in enumerate(bm25_results):
      if isinstance(doc_id, str) and doc_id.isdigit():
        doc_id = int(doc_id)
      bm25_rank[doc_id] = idx + 1

    semantic_results = self.semantic_search.search_chunks(query, internal_limit)
    semantic_rank = {}
    for idx, r in enumerate(semantic_results):
      doc_id = r["doc_id"]
      if isinstance(doc_id, str) and doc_id.isdigit():
        doc_id = int(doc_id)
      semantic_rank[doc_id] = idx + 1

    all_doc_ids = set(bm25_rank.keys()) | set(semantic_rank.keys())

    doc_info = {}
    for doc_id in all_doc_ids:
      doc = self.idx.docmap.get(doc_id, {})
      bm25_r = bm25_rank.get(doc_id)
      sem_r = semantic_rank.get(doc_id)

      score = 0.0
      if bm25_r is not None:
        score += self.rrf_score(bm25_r, k)
      if sem_r is not None:
        score += self.rrf_score(sem_r, k)

      doc_info[doc_id] = {
        "doc": doc,
        "bm25_rank": bm25_r,
        "semantic_rank": sem_r,
        "rrf_score": score,
      }

    sorted_docs = sorted(doc_info.values(), key=lambda x: x["rrf_score"], reverse=True)
    results = []
    if rerank_method == "batch":
      cand_list = sorted_docs[:limit]
      doc_list_str = ""
      for entry in cand_list:
        doc = entry["doc"]
        doc_id = doc.get("id", "")
        doc_list_str += f"- ID: {doc_id}, Title: {doc.get('title', '')}, Description: {doc.get('description', '')[:200]}\n"

      prompt = f"""Rank the movies listed below by relevance to the following search query.
        Query: "{query}"

        Movies:
        {doc_list_str}

        Return the movie IDs in order of relevance, best match first.

        Your response must be a raw JSON array of integers.
        Do not wrap the JSON in Markdown. Do not use a ```json code block.
        Do not include any explanatory text.

        For example:
        [75, 12, 34, 2, 1]

        Ranking:"""

      response = self.get_gemini_response(prompt)

      try:
        ranked_ids = json.loads(response.text.strip())
      except (json.JSONDecodeError, ValueError):
        ranked_ids = []

      doc_id_to_rank = {doc_id: idx for idx, doc_id in enumerate(ranked_ids)}

      for entry in cand_list:
        doc = entry["doc"]
        doc_id = doc.get("id")
        if isinstance(doc_id, str) and doc_id.isdigit():
          doc_id = int(doc_id)
        entry["re_rank_score"] = doc_id_to_rank.get(doc_id, len(ranked_ids))

      cand_list.sort(key=lambda x: x["re_rank_score"])

      results = []
      for entry in cand_list:
        doc = entry["doc"]
        desc = doc.get("description", "")
        if len(desc) > 100:
          desc = desc[:100] + "..."
        results.append({
          "title": doc.get("title", "Unknown"),
          "description": desc,
          "rrf_score": round(entry["rrf_score"], 3),
          "bm25_rank": entry["bm25_rank"],
          "semantic_rank": entry["semantic_rank"],
          "re_rank_score": entry["re_rank_score"],
        })

    elif rerank_method == "individual":
      for entry in sorted_docs[:limit]:
        doc = entry["doc"]
        response = self.llm_base_reranking(query, doc)
        time.sleep(3)

        desc = doc.get("description", "")
        if len(desc) > 100:
          desc = desc[:100] + "..."
        results.append({
          "title": doc.get("title", "Unknown"),
          "description": desc,
          "rrf_score": round(entry["rrf_score"], 3),
          "bm25_rank": entry["bm25_rank"],
          "semantic_rank": entry["semantic_rank"],
          "re_rank_score": response.text,
        })
      results = sorted(results, key=lambda x: x["re_rank_score"], reverse=True)

    else:
      for entry in sorted_docs[:limit]:
        doc = entry["doc"]
        desc = doc.get("description", "")
        if len(desc) > 100:
          desc = desc[:100] + "..."
        results.append({
          "title": doc.get("title", "Unknown"),
          "description": desc,
          "rrf_score": round(entry["rrf_score"], 3),
          "bm25_rank": entry["bm25_rank"],
          "semantic_rank": entry["semantic_rank"],
        })

    if rerank_method == "cross_encoder":
      cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
      pairs = []
      for doc in results:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('description', '')}"])
      
      scores = cross_encoder.predict(pairs)
      encoder_result = []
      for index, entry in enumerate(results):
        encoder_result.append({
          "title": entry.get("title", "Unknown"),
          "description": entry.get("description", "Unknown"),
          "rrf_score": round(entry["rrf_score"], 3),
          "bm25_rank": entry["bm25_rank"],
          "cross_encoder_score": scores[index]
        })
      
      results = sorted(encoder_result, key=lambda x: x["cross_encoder_score"], reverse=True)
      return results
    
    self.gemini_llm(query, enhance)
    if (evaluate):
      for index, result in enumerate(results):
        response  = self.llm_based_evaluation(query, result.get('title'))
        score = json.loads(response.text.strip())[0]
        print(f"{index+1}. {result.get('title')}: {score}/3")
        time.sleep(2)
    return results