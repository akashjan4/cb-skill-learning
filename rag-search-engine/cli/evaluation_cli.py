import argparse
import json 
from pathlib import Path
from lib.hybrid_search import HybridSearch

def main() -> None :
  _base_dir = Path("lib") 
  data_set_path = _base_dir.parent / "data" / "golden_dataset.json"
  
  parser = argparse.ArgumentParser(description="Search Evaluation CLI")
  parser.add_argument("--limit", type=int, default=5,  help="Number of results to evaluate (k for precision@k, recall@k)")
  args = parser.parse_args()
  limit = args.limit
  print(data_set_path)
  with open("data/movies.json", "r", encoding='utf-8') as file:
      documents = json.load(file)
      documents_list = documents.get("movies", [])
      
  with open(data_set_path, 'r', encoding='utf-8') as file:
    data_set = json.load(file)
    golden_data_set = data_set.get("test_cases", [])
  
  hybrid_search = HybridSearch(documents_list)
  
  results = []
  print(f"k={args.limit}")
  for data in golden_data_set:
    k = limit
    result = hybrid_search.rrf_search(data.get("query"), None, None, 60, limit)
    relevant_retrieved = 0
    relevant_docs = data.get('relevant_docs')
    for val in result:
      if val.get('title') in relevant_docs: 
        relevant_retrieved+=1
    
    relevant_score = relevant_retrieved / len(result) # precision   
    recall_score = relevant_retrieved / len(relevant_docs) # precision   
    f1_score = 2 * (relevant_score * recall_score)/(relevant_score + recall_score) # f1 = 2 * (precision * recall) / (precision + recall)
    print(f"- Query: {data.get("query")}")
    print(f"\t- Precision@{k}: {relevant_score:.4f}")
    print(f"\t- Recall@{k}: {recall_score:.4f}")
    print(f"\t- F1 Score: {f1_score:.4f}")
    print(f"\t- Retrieved: {val.get('description')}")
    print(f"\t- Relevant: {", ".join(relevant_docs)}")
    
    results.append(result)
  
if __name__ == "__main__":
  main()