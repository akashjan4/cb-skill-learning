import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search, chunk_text
def main() -> None:
  parser = argparse.ArgumentParser(description="Semantic Search CLI")
  subparsers = parser.add_subparsers(dest="command", help="Available commands")
  
  subparsers.add_parser("verify", help="Verify that the semantic search model loads correctly")
  subparsers.add_parser("verify_embeddings", help="Verify that embeddings can be generated and saved correctly")
  
  embedding_parser = subparsers.add_parser("embed_text", help="Generate embeddings for the input text")
  embedding_parser.add_argument("text", type=str, help="Text to embed")
  
  embed_query_parser = subparsers.add_parser("embed_query", help="Generate embedding for a search query")
  embed_query_parser.add_argument("query", type=str, help="Search query to embed")
  
  search_parser = subparsers.add_parser("search", help="Search for relevant documents based on a query")
  search_parser.add_argument("query", type=str, help="Search query")
  search_parser.add_argument("--limit", type=int, default=5, help="Number of top results to return")

  chunk_parser = subparsers.add_parser("chunk", help="Chunk a long text into smaller pieces")
  chunk_parser.add_argument("text", type=str, help="Text to chunk")
  chunk_parser.add_argument("--chunk-size", type=int, default=512, help="Maximum number of characters per chunk")
  
  args = parser.parse_args()
  match args.command:
    case "verify":
      verify_model()
    case "embed_text":
      embed_text(args.text)
    case "verify_embeddings":
      verify_embeddings()
    case "embed_query":
      embed_query_text(args.query)
    case "search":
      search(args.query, args.limit)
    case "chunk":
      chunk_text(args.text, args.chunk_size)
    case _:
      parser.print_help()

if __name__ == "__main__":
  main()