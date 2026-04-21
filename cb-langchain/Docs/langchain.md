# Comparison: LangChain vs. LangGraph vs. Deep Agents
- Choose LangChain if your project is a straight line where speed and predictability are the priority.
- Choose LangGraph if your project is a flowchart where the AI needs to check its own work or ask for help.
- Choose Deep Agents if your project is a department where different "experts" need to collaborate autonomously.
  
| Consideration | LangChain (LCEL) | LangGraph | Deep Agents (CrewAI/etc.) |
|---------------|-------------------|-----------|--------------------------|
| **Primary Flow** | Linear: A → B → C | Cyclic: Can loop back | Autonomous: Self‑directed |
| **Control** | High: You dictate every step. | Medium: You define the rules/loops. | Low: The agents decide the path. |
| **Reliability** | High: Very predictable. | Medium: Good, but loops can hang. | Variable: Can hallucinate a path. |
| **Cost** | Low: Minimal LLM calls. | Medium: Depends on loop count. | High: Many agents = many calls. |
| **Setup Time** | Fast: Easy to pipe together. | Moderate: Need to define a graph. | Moderate: Need to define roles. |
| **Best For** | Simple RAG & Data Pipelines | Self‑correcting logic & Agents | Complex, multi‑role projects |

# LangChain Expression Language (LCEL)[^1]
- Declarative way to easily compose chains
- Used to build complex LLM pipelines or chains 
- Everything in LCEL is a `Runnable`. Understanding the `RunnablePassthrough`, `RunnableParallel`, and `RunnableLambda` components is 90% of the battle.
- Multiple steps can be chained using vertical bar or pipe operator `|`
  - `|` is an override of python `|` operator 
  - Technically, when python interpreter encounters the `|` operator between two objects, it invokes \_\_or\_\_[^2] method of the object on the right, effectively passing output of previous step as an input of following one. 

- Provide 
  - **Streaming Support** - Enable to get chunks of response tokens with the lowest latency and stream the response live to the user
  - **Async support and parallel execution** - Async api to support concurrent requests in the same server. Certain steps can be executed in parallel if it is supported, LCEL runs them in parallel automatically, reducing latency.
  - Have capability to validate input and output based on the schema 
  - LangServe deployment and LangSmith Observability allowing you to see exactly what happened at every step of the pipe.

- Challenges
  - **Debugging**: When a chain fails, the stack traces can be deep and cryptic because you are debugging the framework's execution engine, not your own simple functions.
  - **Abstraction** Bloat: For a very simple call to an LLM, LCEL can feel like overkill.
  
- Example: 
  - `Chain = prompt | llm | parser`
  - This represents that the __prompt__ flow into the __LLM__ which generates the __response__ which is formatted using the __Parser__ rules.

- Built in Chains 
  - create stuff document chain - Takes list of documents and formats them into a prompt, and pass that prompt to an LLM to generate response 
  - create sql query chain - Generate SQL queries for the given database from natural language 
  - create history aware retriever - Takes conversation history and then use that to generate a rephrased search query if needed.  
  - create retrieval chain - Takes use query and pass it to the retriever to fetch relevant context documents. Then query and context is passed to an LLM to generate response.

- From Release Note: [LangChain releases 1.0 with two major changes:](https://docs.langchain.com/oss/python/langchain/philosophy#2025-10-20)

## Code 
- [example-chain.py](../simple-agent/example-chain.py)
- Sample 
  ```python
  # install langchain and langchina-openai (ollama, anthropic etc)
  from getpass from getpass
  import os 
  from langchain_openai import  ChatOpenAI
  from langchain_core.prompt import ChatPromptTemplate

  OPENAI_KEY = getpass('Please enter your OpenAI key')
  os.environ['OPENAI_API_KEY'] = OPENAI_KEY

  chatgpt = ChatOpenAI(model_name="gpt-<name-of-model>", temperature=0)
  prompt_txt = "{query}"
  prompt_template = ChatPromptTemplate.from_template(prompt_txt)

  llm_chain = (prompt_template | chatgpt)

  response = llm_chain.invoke({'query': 'Explain generative ai to 5 year old. Keep it under 1000 characters'})
  print(response)
  ```
---

## Tracking LLM Cost

- LLMs charges you based on numbers of tokens
- LangChain enables you to track your toke

```python
# Example

prompt = """Explain Generative AI in one line"""

with get_openai_callback() as cbL
  response = chatgpt.invoke(prompt)
  print(response.content)
  print(cb)
  
# Output
# Generative AI is a type of .....
# Tokens Used: 50
  # Prompt Tokens: 15
  # Completion Tokens: 35
# Successful Requests: 1
# Total Cost (USD): $9.25e-05
```

## Caching 

- LangChain provides an optional caching mechanism when interfacing with LLMs
- Saves cost by reducing the number of LLM calls, if you're often requesting the same prompt multiple times
- Response time is much faster when requesting with same prompt
```python
# Example 

chatgpt = ChatOpenAI(model_name="gpt-3.5-turbo",
temperature=0)

# create memory cache
set_llm_cache(InMemoryCache())

# The first time, it is not yet in cache, so it should take longer
prompt = """Explain to me what is mortgage"""
chat_template = ChatPromptTemplate.from_template(prompt)
chatgpt. invoke(chat_template.format())

# Output
# Wall time: 2.27 s
# AIMessage(content='A mortgage is a type of loan....

# Let's now send the same prompt to the LLM
# The cache should give the response back rather than the LLM
# Response time should also be much faster
chatgpt. invoke(chat_template.format())
# Output
# Wall time: 2.23 ms
# AIMessage(content='A mortgage is a type of loan....
```

## Streaming 

- LangChain has its LLM APIs implement the Runnable[^3] interface, which comes with implementation of methods for streaming like stream and astream (async stream)


## Built in Memory Constructs(LCEL) - 
### These methods are now deprecated in newer version of LangChain. LangChain Classic have these methods
- ConversationBufferMemory - ```This memory allows for storing messages of conversation history and then extracts the messages in a variable```
- ConversationBufferWindowMemory - ```Stores only the last K messages of conversation history. This can be useful for keeping a sliding window of the most recent interactions with the LLM```
- ConversationSummaryMemory - ```Summarizes the conversation history and stores the current summary in memory instead of actual conversation messages```
- VectorStoreRetrieverMemory - ```Stores conversation history in a vector database and queries the top-K most relevant docs every time a new message or query is sent to the LLM. Doesn't care about the order of conversation```
- ChatMessageHistory - ```Simple but highly configurable wrapper to store messages in an in-memory list. Easy to manage memory for multiple users and sessions```
- SQLChatMessageHistory - ```Same as ChatMessageHistory but persists conversation in a SQL database. Very useful when managing multiple long conversations```
---
[^1]: [LangChain Expression Language](https://blog.langchain.com/langchain-expression-language/)
[^2]: It is a special method called **dunder method**. When you use the expression `object1 | object2`, Python internally calls object1. `__or__`(object2). The return value of the `__or__` method becomes the result of the expression.
[^3]: https://mirascope.com/blog/langchain-runnables
[^3]: https://reference.langchain.com/python/langchain-core/runnables
[^3]: https://medium.com/@adatiyavinayshaileshbhai/what-is-a-runnable-interface-in-langchain-987991752afa

