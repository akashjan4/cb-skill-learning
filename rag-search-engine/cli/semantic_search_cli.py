import argparse
from pathlib import Path
import json
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search, chunk_text, semantic_chunk
from lib.chunked_semantic_search import ChunkedSemanticSearch
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
  chunk_parser.add_argument("--chunk-size", type=int, help="Maximum number of characters per chunk")
  chunk_parser.add_argument("--overlap", type=int, help="Number of overlapping characters between chunks")

  semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk text based on semantic similarity (not implemented yet)")
  semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk semantically")
  semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Maximum number of characters per chunk")
  semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping characters between chunks")


  subparsers.add_parser("embed_chunks", help="embed you large docs")
  
  search_chunked_parser = subparsers.add_parser("search_chunked")
  search_chunked_parser.add_argument("text", type=str)
  search_chunked_parser.add_argument("--limit", type=int)
  
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
      chunk_text(args.text, args.chunk_size, args.overlap)
    case "semantic_chunk":
      semantic_chunk(args.text, args.max_chunk_size, args.overlap)
    case "embed_chunks":
      chunk_sem_search = ChunkedSemanticSearch()
      with open("data/movies.json", "r", encoding='utf-8') as file:
        documents = json.load(file)
      documents_list = documents.get("movies", [])
      embeddings = chunk_sem_search.load_or_create_chunk_embeddings(documents_list)
      print(f"Generated {len(embeddings)} chunked embeddings")
    case "search_chunked":
      chunk_sem_search = ChunkedSemanticSearch()
      with open("data/movies.json", "r", encoding='utf-8') as file:
        documents = json.load(file)
      documents_list = documents.get("movies", []) # load your movies
      chunk_sem_search.load_or_create_chunk_embeddings(documents_list)
      chunk_sem_search.search_chunks(args.text, args.limit)
    case _:
      parser.print_help()

if __name__ == "__main__":
  main()