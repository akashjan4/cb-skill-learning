import os
from ollama import chat
from colorama import Fore, Style

model = "gemma4:latest"
generation_chat_history = [
  {
      "role": "system",
      "content": "You are a financial advisor. You analyze the stock market and provide suggestions to investors"
      "in which stock is ready to buy and which can be sold."
      "You have access to the stock market data and you can analyze it to provide insights to investors."
      "Keep in mind the provided information must be true, accurate and up to date. Don't make up any information. If you don't know the answer, say you don't know."
      "If investor provides critique, respond with a revised version of your previous response that addresses the critique."
      "Don't use emojis in your response.",
  }
]

generation_chat_history.append(
  {"role": "user", "content": "What is the right to invest in Ather Energy stock?"}
)

reflection_chat_history = [
  {
      "role": "system",
      "content": "you are SEBI-registered financial advisor. "
      "You tasked with generating critique and recommendations "
      "for the user's analysis. If you evaluate the user's analysis to be good, "
      "then simply output keyword: <OK>",
  }
]


def add_to_history(history_collection, role, content):
  """Helper function to add messages to the history collection"""
  history_collection.append({"role": role, "content": content})


def get_analysis_response(message_history):
  """Helper function to get response from the model based on the message history"""
  response = chat(model=model, messages=message_history).message.content
  return response

def fancy_print(message):
  """Helper function to print messages in a fancy format"""
  print(Style.BRIGHT + Fore.CYAN + f"\n{'='* 50}")
  print(Fore.MAGENTA + f"{message}")
  print(Style.BRIGHT + Fore.CYAN + f"\n{'='* 50}\n")

def main():
  critique_response = None
  step = 0
  while critique_response is None or "<OK>" not in critique_response.strip():
    analysis_response = get_analysis_response(generation_chat_history)
    add_to_history(generation_chat_history, "assistant", analysis_response)

    critique_response = get_analysis_response(reflection_chat_history)
    add_to_history(reflection_chat_history, "user", analysis_response)
    add_to_history(generation_chat_history, "user", critique_response)
    
    step += 1
    
    fancy_print(f"Step: {step}")
    fancy_print(f"Analysis Response:\n{analysis_response}")
    fancy_print(f"Critique Response:\n{critique_response}")

if __name__ == "__main__":
  main()
