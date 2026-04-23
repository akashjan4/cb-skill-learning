from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_template(
  "Tell me a {category} about {topic}. Add emoji whenever you feel it will add value"
)

output_parser = StrOutputParser()
model = ChatOllama(model="gpt-oss:20b")

chain = (
  {
    "category": RunnablePassthrough(), 
    "topic": RunnablePassthrough() 
  }
  |prompt
  |model
  |output_parser
)

response = chain.invoke({"category": "news", "topic": "ice cream"})
print(response)
