class ChatHistory(list):
  def __init__(self, messages: list | None, total_length: int = -1) :
    super().__init__(messages or [])
    self.total_length = total_length

  def append(self, message: str):
    if self.total_length > 0 and len(self) >= self.total_length:
      self.pop(1)
    super().append(message)
