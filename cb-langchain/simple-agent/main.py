from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy


SYSTEM_PROMPT ="""You are an expert weather forecaster, who speaks in a helpful tone.

You have access to two tools:
- get_weather_for_location: use this to get the weather for specific locations.
- get_user_location: use this to get the user's location.

If a user asks you about the weather, make sure you know the location. If user can tell from the question that they mean 
wherever they are, use the get_user_location."""




@dataclass
class Context:
  """Customer runtime context schema."""
  user_id: str


@tool
def get_weather_for_location(location: str) -> str:
  """Get weather for a given location."""
  return f"It's always sunny in {location}!"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
  """Get the user's location."""
  user_id = runtime.context.user_id
  return "Florida" if user_id == "1" else "SF"

model = init_chat_model("ollama:gpt-oss:20b", temperature=0.5)

@dataclass 
class ResponseFormat:
  """Response format schema."""
  punny_response: str
  weather_conditions: str | None = None

checkpointer = InMemorySaver()
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

config = {"configurable": {"thread_id": "1"}}



# def get_weather(city: str) -> str:
#   """Get weather for a given city."""
#   print(f"Getting weather for {city}...")
#   return f"It's always sunny in {city}!"

# agent = create_agent(
#   model="ollama:gpt-oss:20b",
#   tools=[get_weather],
#   system_prompt="You are a horney assistant",
# )

# # Run the agent
# response =agent.invoke(
#   {"messages": [{"role": "user", "content": "what is the weather in new york?"}]}
# )

# print(response["messages"][-1].content)
def main():
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
        config=config,
        context=Context(user_id="1")
    )

    print(response['structured_response'])


    response = agent.invoke(
        {"messages": [{"role": "user", "content": "thank you!"}]},
        config=config,
        context=Context(user_id="1")
    )


    print(response['structured_response'])


if __name__ == "__main__":
    main()
