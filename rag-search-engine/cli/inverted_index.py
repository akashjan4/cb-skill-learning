import math
import json
import re
import pickle
from pathlib import Path
from collections import Counter
from nltk.stem import PorterStemmer

class InvertedIndex:
  # Constants moved to class level for easy access
  BM25_K1 = 1.5
  BM25_B = 0.75

  def __init__(self):
    self.index = {}
    self.docmap = {}
    self.term_frequency = {}
    self.doc_lengths = {}

    self._base_dir = Path(__file__).resolve().parents[1]
    self._data_path = self._base_dir / "data" / "movies.json"
    self._stopwords_path = self._base_dir / "data" / "stopwords.txt"
    self._cache_dir = self._base_dir / "cache"
    self._index_path = self._cache_dir / "index.pkl"
    self._docmap_path = self._cache_dir / "docmap.pkl"
    self._tf_path = self._cache_dir / "term_frequency.pkl"
    self._doc_lengths_path = self._cache_dir / "doc_lengths.pkl"

    self._stemmer = PorterStemmer()
    self.stop_words = self.__get_stop_words()
    self._load_cache()

  def __get_stop_words(self):
    try:
      with open(self._stopwords_path, "r", encoding='utf-8') as file:
        return set(file.read().splitlines())
    except FileNotFoundError:
      return set()

  def _load_cache(self):
    if self._index_path.exists():
      with open(self._index_path, "rb") as f:
        self.index = pickle.load(f)
    if self._docmap_path.exists():
      with open(self._docmap_path, "rb") as f:
        self.docmap = pickle.load(f)
    if self._tf_path.exists():
      with open(self._tf_path, "rb") as f:
        self.term_frequency = pickle.load(f)
    if self._doc_lengths_path.exists():
      with open(self._doc_lengths_path, "rb") as f:
        self.doc_lengths = pickle.load(f)

  def _decode_unicode_escapes(self, text: str) -> str:
    # Decode only literal double-escaped unicode sequences (\\uXXXX)
    # to avoid corrupting standard characters (like smart quotes) in the dataset.
    return re.sub(
        r'\\u([0-9a-fA-F]{4})',
        lambda m: chr(int(m.group(1), 16)),
        text
    )

  def __tokenize(self, text):
    normalized_text = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = [token for token in normalized_text.split() if token]
    # Filter stop words and stem
    tokens = [self._stemmer.stem(q) for q in tokens if q.lower() not in self.stop_words]
    return tokens

  def __add_document(self, doc_id, text):
    tokens = self.__tokenize(text)
    # Ensure doc_id is consistent (integer)
    doc_id = int(doc_id)

    counter = self.term_frequency.setdefault(doc_id, Counter())
    for token in tokens:
      counter[token] += 1
      if token not in self.index:
        self.index[token] = set()
      self.index[token].add(doc_id)

    # Store the length of the document (number of tokens)
    self.doc_lengths[doc_id] = len(tokens)

  def build(self):
    self.index = {}
    self.docmap = {}
    self.term_frequency = {}
    self.doc_lengths = {}

    with open(self._data_path, "r", encoding='utf-8') as file:
      movies = json.load(file)

    for movie in movies.get("movies", []):
      doc_id = movie.get("id")
      if doc_id is None:
        continue

      title = self._decode_unicode_escapes(movie.get("title", ""))
      description = self._decode_unicode_escapes(movie.get("description", ""))
      
      # Update movie dictionary with clean decoded text
      movie["title"] = title
      movie["description"] = description

      content = f"{title} {description}"
      self.__add_document(doc_id, content)
      self.docmap[doc_id] = movie
    self.save()

  def save(self):
    self._cache_dir.mkdir(parents=True, exist_ok=True)
    with open(self._index_path, "wb") as f:
      pickle.dump(self.index, f)
    with open(self._docmap_path, "wb") as f:
      pickle.dump(self.docmap, f)
    with open(self._tf_path, "wb") as f:
      pickle.dump(self.term_frequency, f)
    with open(self._doc_lengths_path, "wb") as f:
      pickle.dump(self.doc_lengths, f)

  def rebuild(self):
    print("Clearing cache and rebuilding...")
    self.index = {}
    self.docmap = {}
    self.term_frequency = {}
    self.doc_lengths = {}
    self.build()

  def get_documents(self, query: str) -> list:
    """Returns a list of document IDs that contain any query terms,
    sorted by how many query terms they match (most to least)."""
    tokens = self.__tokenize(query)
    if not tokens:
      return []

    doc_counts = Counter()
    for token in tokens:
      for doc_id in self.index.get(token, []):
        doc_counts[doc_id] += 1

    return [doc_id for doc_id, count in doc_counts.most_common()]

  def calculate_tf(self, doc_id, term: str) -> float:
    """Calculates term frequency normalized by document length (tf = count / doc_len)."""
    tokens = self.__tokenize(term)
    if not tokens:
      return 0.0
    stemmed_term = tokens[0]
    doc_id = int(doc_id)

    count = self.term_frequency.get(doc_id, {}).get(stemmed_term, 0)
    doc_len = self.doc_lengths.get(doc_id, 0)
    if doc_len == 0:
      return 0.0
    return count / doc_len

  def calculate_idf(self, term: str) -> float:
    """Calculates standard Inverse Document Frequency: idf = log(N / DF)."""
    tokens = self.__tokenize(term)
    if not tokens:
      return 0.0
    stemmed_term = tokens[0]

    doc_count = len(self.docmap)
    if doc_count == 0:
      return 0.0

    doc_freq = len(self.index.get(stemmed_term, []))
    if doc_freq == 0:
      return 0.0

    return math.log(doc_count / doc_freq)

  def calculate_tf_idf(self, doc_id, term: str) -> float:
    """Calculates TF-IDF score: tf * idf."""
    tf = self.calculate_tf(doc_id, term)
    idf = self.calculate_idf(term)
    return tf * idf

  def __get_avg_doc_length(self) -> float:
    if not self.doc_lengths:
      return 0.0
    return sum(self.doc_lengths.values()) / len(self.doc_lengths)

  def get_bm25_idf(self, term: str) -> float:
    # Stem term to ensure consistency with index keys
    tokens = self.__tokenize(term)
    stemmed_term = tokens[0] if tokens else term

    doc_count = len(self.docmap)
    doc_freq = len(self.index.get(stemmed_term, []))
    # Standard BM25 IDF formula (Okapi version, without +1 smoothing)
    return math.log((doc_count - doc_freq + 0.5) / (doc_freq + 0.5))

  def get_bm25_tf(self, doc_id, term: str, k1=None, b=None) -> float:
    if k1 is None:
      k1 = self.BM25_K1
    if b is None:
      b = self.BM25_B

    tokens = self.__tokenize(term)
    stemmed_term = tokens[0] if tokens else term

    tf = self.term_frequency.get(int(doc_id), {}).get(stemmed_term, 0)
    doc_len = self.doc_lengths.get(int(doc_id), 0)
    avg_doc_len = self.__get_avg_doc_length()

    # BM25 normalization component
    numerator = tf * (k1 + 1)
    denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
    return numerator / denominator if denominator != 0 else 0

  def bm25_search(self, query: str, limit=10, k1=None, b=None) -> list:
    tokens = self.__tokenize(query)
    if not tokens:
      return []

    scores = {}
    for token in tokens:
      idf = self.get_bm25_idf(token)
      for doc_id in self.index.get(token, []):
        tf_score = self.get_bm25_tf(doc_id, token, k1=k1, b=b)
        scores[doc_id] = scores.get(doc_id, 0.0) + (tf_score * idf)

    sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_docs[:limit]
