import argparse
import json


def main() -> None:
  parser = argparse.ArgumentParser(description="Hybrid Search CLI")
  subparsers = parser.add_subparsers(dest="command", help="Available commands")

  ws_parser = subparsers.add_parser("weighted-search", help="Weighted hybrid search combining BM25 and semantic scores")
  ws_parser.add_argument("query", type=str, help="Search query")
  ws_parser.add_argument("--alpha", type=float, default=0.5, help="Weight for semantic score (0 = pure BM25, 1 = pure semantic)")
  ws_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return")

  rrf_parser = subparsers.add_parser("rrf-search", help="Reciprocal Rank Fusion hybrid search")
  rrf_parser.add_argument("query", type=str, help="Search query")
  rrf_parser.add_argument("-k", type=int, default=60, help="RRF constant to smooth rank scores")
  rrf_parser.add_argument("--limit", type=int, default=5, help="Maximum number of results to return")
  rrf_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="spelling correction to improve queries before running hybrid search." )
  rrf_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"])
  rrf_parser.add_argument("--evaluate", action="store_true" , help="LLM Evaluation" )
  
  normalize_parser = subparsers.add_parser("normalize", help="Min-max normalize a list of scores")
  normalize_parser.add_argument("scores", type=float, nargs="*", help="Scores to normalize")

  args = parser.parse_args()
  match args.command:
    case "weighted-search" | "rrf-search":
      from lib.hybrid_search import HybridSearch

      with open("data/movies.json", "r", encoding='utf-8') as file:
        documents = json.load(file)
      documents_list = documents.get("movies", [])

      hybrid = HybridSearch(documents_list)

      if args.command == "weighted-search":
        results = hybrid.weighted_search(args.query, alpha=args.alpha, limit=args.limit)
      else:
        results = hybrid.rrf_search(args.query, args.enhance, args.rerank_method, args.evaluate, k=args.k, limit=args.limit)

      if not results:
        print("No results found.")
      else:
        for i, r in enumerate(results):
          print(f"{i+1}. {r['title']}")
          if "rrf_score" in r:
            bm25_rank = r.get("bm25_rank") or "-"
            sem_rank = r.get("semantic_rank") or "-"
            re_rank_score = r.get("re_rank_score") or ""
            cross_encoder_score = r.get("cross_encoder_score") or ""
            if re_rank_score:
              print(f"  Re-rank Score: {re_rank_score}")
            if cross_encoder_score:
              print(f"  Cross Encoder Score: {cross_encoder_score}")
            print(f"  RRF Score: {r['rrf_score']:.3f}")
            print(f"  BM25 Rank: {bm25_rank}, Semantic Rank: {sem_rank}")
         
          else:
            print(f"  Hybrid Score: {r['hybrid_score']:.3f}")
            print(f"  BM25: {r['bm25_score']:.3f}, Semantic: {r['semantic_score']:.3f}")
          print(f"  {r['description']}")

    case "normalize":
      if not args.scores:
        return
      mn = min(args.scores)
      mx = max(args.scores)
      if mn == mx:
        for _ in args.scores:
          print(1.0)
      else:
        for s in args.scores:
          print((s - mn) / (mx - mn))
    case _:
      parser.print_help()

if __name__ == "__main__":
  main()