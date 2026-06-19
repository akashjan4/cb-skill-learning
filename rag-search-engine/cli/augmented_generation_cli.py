import argparse
from lib.hybrid_search import HybridSearch
import json
import os
from dotenv import load_dotenv
from google import genai


load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
  raise RuntimeError("GEMINI_API_KEY environment variable not set")

gemma_client = genai.Client(api_key=api_key)

def get_gemini_response(prompt):
  return gemma_client.models.generate_content(
    model="gemma-4-31b-it",
    contents=prompt
  )
  
def llm_augment_generation(query, docs):
  prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.

    Query: {query}

    Documents:
    {docs}

    Answer:"""

  return get_gemini_response(prompt)

def llm_summary_generation(query, results):
  prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.
    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.
    Query: {query}

    Search results:
    {results}

    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
  
  return get_gemini_response(prompt)

def llm_summary_citation(query, documents):
  prompt = f"""Answer the query below and give information based on the provided documents.

  The answer should be tailored to users of Hoopla, a movie streaming service.
  If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

  Query: {query}

  Documents:
  {documents}

  Instructions:
  - Provide a comprehensive answer that addresses the query
  - Cite sources in the format [1], [2], etc. when referencing information
  - If sources disagree, mention the different viewpoints
  - If the answer isn't in the provided documents, say "I don't have enough information"
  - Be direct and informative

  Answer:"""
  
  return get_gemini_response(prompt)

def __get_movie_docs():
  with open("data/movies.json", "r", encoding='utf-8') as file:
    documents = json.load(file)
    documents_list = documents.get("movies", [])
  return documents_list

def __get_movie_search_result(query, k=60, limit=5):
  documents_list = __get_movie_docs()
  hybrid_search = HybridSearch(documents_list)
  return hybrid_search.rrf_search(query, None, None, k, limit)
 
  
def main() -> None:
  parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
  subparsers = parser.add_subparsers(dest="command", help="Available commands")

  rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
  rag_parser.add_argument("query", type=str, help="Search query for RAG")
  
  summary_parser = subparsers.add_parser("summarize", help="summarize the results")
  summary_parser.add_argument("query", type=str, help="query text")
  summary_parser.add_argument("--limit", type=int, help="set limit")
  
  citation_parser = subparsers.add_parser("citations", help="add citations to generated summary for adding credibility")
  citation_parser.add_argument("query", type=str, help="query text")
  citation_parser.add_argument("--limit", type=int, help="set limit")

  args = parser.parse_args()

  match args.command:
    case "rag":
      query = args.query
      results = __get_movie_search_result(query)
      response = llm_augment_generation(query, results)
      print("Search Results:")
      for res in results:
        print(f"- {res.get('title')}")
      
      print("\nRAG Response:")
      print(response.text)
    case "summarize":
      query = args.query
      results = __get_movie_search_result(query)
      response = llm_summary_generation(query, results)
      for res in results:
        print(f"- {res.get('title')}")
      print("\nLLM Summary:")
      print(response.text)
    case "citations":
      query = args.query
      docs = __get_movie_docs()
      results = __get_movie_search_result(query, limit=15)
      response = llm_summary_citation(query, results)
      for res in results:
        print(f"- {res.get('title')}")
      print("\nLLM Summary:")
      print(response.text)
    case _:
      parser.print_help()

if __name__ == "__main__":
  main()