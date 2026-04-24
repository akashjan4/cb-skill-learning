from chat_history import ChatHistory

class FixedFirstChatHistory(ChatHistory):
  def __init__(self, messages: list | None, total_length: int = -1) :
    super().__init__(messages, total_length)
  
  def append(self, message: str):
    if len(self) == self.total_length:
      self.pop(1)
    super().append(message)