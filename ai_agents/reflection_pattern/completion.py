from ollama import chat

def completion_create(history: list, model: str):
  response = chat(model=model, messages=history).message.content
  return response

def build_prompt_structure(prompt: str, role: str, tag: str = None) -> dict:
  if tag:
    prompt = f"<{tag}>\n{prompt}\n</{tag}>"
  return {"role": role, "content": prompt}

def update_chat_history(history: list, message: str, role: str):
  history.append({"role": role, "content": message})