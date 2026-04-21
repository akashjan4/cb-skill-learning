from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.schema.runnable import RunnableLambda, RunnablePassthrough
import pprint

research_abstract_file_name = "research_abstract.txt"
system_prompt_file_name = "system_prompt.txt"
llm = ChatOllama(model="gemma4:latest")

sys_prompt_txt = open(system_prompt_file_name, "r").read()
research_abstract = open(research_abstract_file_name, "r").read()

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", sys_prompt_txt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{instruction}"),
    ]
)

messages = []
memory = ConversationBufferMemory(return_messages=True)
memory.load_memory_variables({})


def get_memory_messages(query):
    return memory.load_memory_variables(query)["history"]


chain = (
    RunnablePassthrough.assign(history=RunnableLambda(get_memory_messages))
    | chat_prompt
    | llm
)


def report_for_general_audience(research_abstract: str) -> str:
    return f"""
    Base on the following research paper abstract,
    create a summary report of maximum 10 lines for 
    a general audience.

    Abstract: {research_abstract}
    """


def report_for_robotics_company() -> str:
    return """
   Use only the research paper abstract from earlier and create a detailed report for a robotics company.
   In report, also include bullet points (3 max) for pros and cons of ethics in robotics AI.
   """


def report_for_gen_ai() -> str:
    return """
    Use only the research paper abstract from earlier and create a detailed report for a generative AI company solving robotics problem.
    In report, also include sections for key points mentioned around collaborative robots.
   """


def invoke_chain(instruction: str):
    user_instruction = {"instruction": instruction}
    response = chain.invoke(user_instruction)
    memory.save_context({"input": instruction}, {"output": response.content})


def main():
    print("Hello from research-paper-analyst!")
    invoke_chain_sequence = [
        report_for_general_audience(research_abstract),
        report_for_robotics_company(),
        report_for_gen_ai(),
    ]

    for invoke_chain_seq in invoke_chain_sequence:
        invoke_chain(invoke_chain_seq)
    print("//----------------START-------------//")
    print("\n")
    pprint.pp(get_memory_messages("history"))
    print("\n")
    print("//-------------END----------------//")


if __name__ == "__main__":
    main()
