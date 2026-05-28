import argparse
from inverted_index import InvertedIndex

def main() -> None:
  parser = argparse.ArgumentParser(description="Keyword Search CLI")
  subparsers = parser.add_subparsers(dest="command", help="Available commands")

  # Search command
  search_parser = subparsers.add_parser("search", help="Search movies using boolean inverted index matching")
  search_parser.add_argument("query", type=str, help="Search query")
  search_parser.add_argument("--limit", type=int, default=10, help="Maximum number of results to return")

  # Build commands
  subparsers.add_parser("build", help="Build movie index using inverted index")
  subparsers.add_parser("rebuild", help="Rebuild movie index using inverted index")

  # Standard metrics commands
  tf_parser = subparsers.add_parser("tf", help="Term Frequency for a given term in a document")
  tf_parser.add_argument("doc_id", type=int, help="Document ID")
  tf_parser.add_argument("term", type=str, help="Term to calculate frequency for")

  idf_parser = subparsers.add_parser("idf", help="Inverse Document Frequency for a given term")
  idf_parser.add_argument("term", type=str, help="Term to calculate IDF for")

  tfidf_parser = subparsers.add_parser("tfidf", help="TF-IDF score for a given term in a document")
  tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
  tfidf_parser.add_argument("term", type=str, help="Term to calculate TF-IDF for")

  # BM25 metrics commands
  bm25idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
  bm25idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

  bm25tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
  bm25tf_parser.add_argument("doc_id", type=int, help="Document ID")
  bm25tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
  bm25tf_parser.add_argument("--k1", type=float, default=None, help="Tunable BM25 K1 parameter")
  bm25tf_parser.add_argument("--b", type=float, default=None, help="Tunable BM25 b parameter")

  # BM25 search command
  bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
  bm25search_parser.add_argument("query", type=str, help="Search query")
  bm25search_parser.add_argument("--k1", type=float, default=None, help="Tunable BM25 K1 parameter")
  bm25search_parser.add_argument("--b", type=float, default=None, help="Tunable BM25 b parameter")
  bm25search_parser.add_argument("--limit", type=int, default=10, help="Maximum number of results to return")

  args = parser.parse_args()
  inverted_index_search = InvertedIndex()

  match args.command:
    case "search":
      results = inverted_index_search.get_documents(args.query)
      if not results:
        print("No movies found matching the query.")
      else:
        for index, result in enumerate(results[:args.limit]):
          title = inverted_index_search.docmap.get(result, {}).get("title", "Unknown")
          print(f"{index+1}.  {title}")

    case "build":
      inverted_index_search.build()
      print("Index built successfully.")

    case "rebuild":
      inverted_index_search.rebuild()
      print("Index rebuilt successfully.")

    case "tf":
      tf = inverted_index_search.calculate_tf(args.doc_id, args.term)
      print(f"Term frequency of '{args.term}' in document '{args.doc_id}': {tf:.6f}")

    case "idf":
      idf = inverted_index_search.calculate_idf(args.term)
      print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

    case "tfidf":
      tf_idf = inverted_index_search.calculate_tf_idf(args.doc_id, args.term)
      print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.6f}")

    case "bm25idf":
      bm25idf = inverted_index_search.get_bm25_idf(args.term)
      print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

    case "bm25tf":
      bm25tf = inverted_index_search.get_bm25_tf(args.doc_id, args.term, k1=args.k1, b=args.b)
      print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

    case "bm25search":
      results = inverted_index_search.bm25_search(args.query, limit=args.limit, k1=args.k1, b=args.b)
      if not results:
        print("No movies found matching the query.")
      else:
        for index, (doc_id, score) in enumerate(results):
          doc = inverted_index_search.docmap.get(doc_id, {})
          print(f"{index+1}.  ({doc.get('id', 'Unknown')}) {doc.get('title', 'Unknown')} - Score: {score:.2f}")

    case _:
      parser.print_help()

if __name__ == "__main__":
  main()
