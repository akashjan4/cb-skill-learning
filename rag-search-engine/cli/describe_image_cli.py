import argparse
import mimetypes
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
gemma_client = genai.Client(api_key=api_key)

def __get_gemini_response(prompt):
  return gemma_client.models.generate_content(
    model="gemma-4-31b-it",
    contents=prompt
  )
  
def __get_mime_type(path):
  mime, _ = mimetypes.guess_type(path)
  mime = mime or "image/jpeg"
  return mime

def explain_image(img, mime, query):
  system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary"""
  parts = [
    system_prompt,
    genai.types.Part.from_bytes(data=img, mime_type=mime),
    query.strip(),
  ]
  return __get_gemini_response(parts)

def main()-> None:
  parser = argparse.ArgumentParser(description="image description CLI")
  parser.add_argument("--image", type=str, help="the path to an image file")
  parser.add_argument("--query", type=str, help="the text query to rewrite based on the image")

  args = parser.parse_args()
  image_mime = __get_mime_type(args.image)
  
  with open(args.image, "rb") as file:
    image_bytes = file.read()
    
  response = explain_image(image_bytes, image_mime, args.query)
  print(f"Rewritten query: {response.text.strip()}")
  if response.usage_metadata is not None:
      print(f"Total tokens:    {response.usage_metadata.total_token_count}")
  
if __name__ == "__main__":
  main()