import argparse
from lib.multimodal_search import verify_image_embedding, image_search_command

def image_search(image_path: str):
  results = image_search_command(image_path)
  for i, res in enumerate(results, 1):
    desc = res['description']
    # Format description exactly as requested
    if len(desc) > 100:
      desc = desc[:100] + "..."
      
    print(f"{i}. {res['title']} (similarity: {res['similarity']:.3f})")
    print(f"   {desc}")
    print()

def main():
  parser = argparse.ArgumentParser(description="image and text Search CLI")
  subparsers = parser.add_subparsers(dest="command", help="Available commands")
  
  img_parser = subparsers.add_parser("verify_image_embedding")
  img_parser.add_argument("image_path", type=str, help="image path")
  
  img_search_parser = subparsers.add_parser("image_search")
  img_search_parser.add_argument("image_path", type=str, help="image path")
  
  args = parser.parse_args()
  
  match args.command:
    case "verify_image_embedding":
      verify_image_embedding(args.image_path)
    case "image_search":
      image_search(args.image_path)
    case _:
      parser.print_help()
   
if __name__ == "__main__":
  main()