# 🔍 RAG Search Engine: Inverted Index & BM25 Search CLI

## 📖 Core Concepts: How it Works

### 1. What is an Inverted Index?
Imagine you have a library of 5,000 books, and you want to find every book that mentions the word "detective". Without an index, you would have to scan every page of every book—this is called a **full-text scan** and is extremely slow ($O(N)$ complexity, where $N$ is the corpus size).

An **Inverted Index** solves this by flipping the search direction. It is like the index at the back of a textbook: it maps every unique **word (term)** to a list of **document IDs** that contain it.
```
Term       ->  Postings List (Document IDs)
-------------------------------------------
"toy"      ->  {1562, 4287, 735}
"stori"    ->  {1562, 2061, 4188}
"detect"   ->  {12, 104, 1562, 3880}
```
At query time, if you search for "toy", the search engine looks up `"toy"` in the index and instantly gets `{1562, 4287, 735}` in $O(1)$ constant time.

### 2. Natural Language Preprocessing (NLP)
Before words are added to the index, they must be cleaned and normalized so that similar terms match each other (solving the "vocabulary mismatch" problem):
*   **Case Normalization**: All text is converted to lowercase so that "Story" and "story" are indexed as the same term.
*   **Punctuation Stripping**: Characters like commas, periods, and quotes are removed.
*   **Stopword Filtering**: Extremely common words (like *the*, *is*, *and*, *in*) are filtered out using `data/stopwords.txt` because they carry no unique meaning and would bloat the index size.
*   **Porter Stemming**: Words are reduced to their root forms (e.g. *stories*, *storying*, and *story* all become **`stori`**; *loved*, *loving*, and *love* all become **`love`**). This ensures that a query for "loving" matches a document containing "loved".

---

## 🧮 Understanding relevance: TF-IDF vs. BM25

Once we find the documents containing our query terms, we need to rank them. We do this by calculating a score.

### Method A: Standard TF-IDF
Traditional TF-IDF ranks documents based on two factors:
1.  **Term Frequency (TF)**: How frequently does the term appear in the document? In our code:
    $$\text{TF}(t, d) = \frac{\text{count of term } t \text{ in document } d}{\text{total tokens in document } d}$$
2.  **Inverse Document Frequency (IDF)**: How rare is the term across the entire corpus?
    $$\text{IDF}(t) = \ln\left(\frac{N}{\text{DF}(t)}\right)$$
    If a term is rare (low Document Frequency $\text{DF}$), its IDF is high. If a term is common (like "movie"), its IDF is low.

### Method B: Okapi BM25 (Best Matching 25)
BM25 is a more advanced ranking function that improves upon TF-IDF by addressing two major flaws of TF-IDF:

#### Flaw 1: Term Frequency Saturation ($k_1$)
In TF-IDF, if "love" appears 20 times in a document, it gets 20 times the TF weight of a document where it appears once. In reality, a document mentioning "love" 20 times isn't 20 times more relevant than one mentioning it 5 times.
*   **BM25 Solution**: It uses a parameter $k_1$ to **saturate** term frequency. As the frequency increases, the score asymptotes.
*   **Formula**:
    $$\text{TF}_{\text{BM25}} = \frac{f \cdot (k_1 + 1)}{f + k_1 \cdot \text{length\_norm}}$$
    *(Where $f$ is the term frequency, and $k_1$ is typically set to $1.5$.)*

#### Flaw 2: Document Length Normalization ($b$)
Longer documents naturally have more words, so they have a higher chance of repeating terms. BM25 penalizes long documents and rewards shorter ones.
*   **BM25 Solution**: It scales the penalty using parameter $b$ (typically set to $0.75$).
*   **Formula**:
    $$\text{length\_norm} = (1 - b) + b \cdot \left(\frac{\text{document length}}{\text{average document length}}\right)$$
    If $b = 1$, we apply full document length normalization. If $b = 0$, document length normalization is completely disabled.

### The Complete Okapi BM25 Formula
For each term $q_i$ in a query:
$$\text{Score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where the Okapi IDF is:
$$\text{IDF}(q_i) = \ln\left(\frac{N - \text{DF}(q_i) + 0.5}{\text{DF}(q_i) + 0.5}\right)$$

---

## 🛠️ System Architecture & Implementation

Here are the key files and methods implemented in the [rag-search-engine/cl/](file://cb-skill-learning/rag-search-engine/cli) directory:

### 1. [inverted_index.py](file:///cb-skill-learning/rag-search-engine/cli/inverted_index.py)
This is the core search engine class.
*   `build(self)`: Parses `data/movies.json`, cleans and tokenizes titles and descriptions, registers mappings, and saves pickle files to `cache/` so we don't have to rebuild the index every time we run a search query.
*   `_decode_unicode_escapes(self, text)`: Uses regex to decode double-escaped characters (e.g. `L\\u00e8vres` to `Lèvres`) prior to indexing. This prevents characters with accents from splitting into separate junk tokens, maintaining exact document statistics.
*   `get_documents(self, query)`: Performs boolean index matching, returning a list of documents matching at least one query term, sorted by the number of matching query terms.
*   `bm25_search(self, query, limit, k1, b)`: Implements the full BM25 ranking algorithm, returning the top $K$ results sorted by relevance.

### 2. [keyword_search_cli.py](file:///cb-skill-learning/rag-search-engine/cli/keyword_search_cli.py)
This is the Command Line Interface. It parses commands, converts inputs, and delegates processing directly to `InvertedIndex`.

---

## 🚀 How to Run the Project

Ensure you have [uv](https://docs.astral.sh/uv/) installed.

1.  **Install dependencies**:
    ```bash
    uv sync
    ```
2.  **Build the index**:
    ```bash
    .venv/bin/python cli/keyword_search_cli.py build
    ```
3.  **Run a ranked BM25 search**:
    ```bash
    .venv/bin/python cli/keyword_search_cli.py bm25search "love story" --limit 5
    ```

### Other Metric Lookups (Great for Debugging)
*   **Relative Term Frequency**: `.venv/bin/python cli/keyword_search_cli.py tf 1562 toy`
*   **Inverse Document Frequency**: `.venv/bin/python cli/keyword_search_cli.py idf toy`
*   **TF-IDF Score**: `.venv/bin/python cli/keyword_search_cli.py tfidf 1562 toy`
*   **BM25 IDF Score**: `.venv/bin/python cli/keyword_search_cli.py bm25idf toy`
*   **BM25 TF Score**: `.venv/bin/python cli/keyword_search_cli.py bm25tf 1562 toy`