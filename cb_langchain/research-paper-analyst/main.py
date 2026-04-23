from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

research_abstract_file_name = "research_abstract.txt"
system_prompt_file_name = "system_prompt.txt"
llm = ChatOllama(model="gemma4:latest")

sys_prompt_txt = open(system_prompt_file_name, "r").read()
research_abstract = open(research_abstract_file_name, "r").read()

chat_prompt = ChatPromptTemplate.from_messages(
    [("system", sys_prompt_txt), ("human", "{instruction}")]
)

chain = chat_prompt | llm
messages = []

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


def report_for_gen_ai ()-> str:
    return """
    Use only the research paper abstract from earlier and create a detailed report for a generative AI company solving robotics problem.
    In report, also include sections for key points mentioned around collaborative robots."""

def invoke_chain(instruction: str):
    messages.append(HumanMessage(content=instruction))
    print(len(messages))
    user_instruction = {"instruction": messages}
    response = chain.invoke(user_instruction)
    messages.append(response)
    return response.content
 
def main():
    print("Hello from research-paper-analyst!")
    invoke_chain_sequence = [report_for_general_audience(research_abstract), report_for_robotics_company(), report_for_gen_ai()]

    for invoke_chain_seq in invoke_chain_sequence:
        response = invoke_chain(invoke_chain_seq)
        print("//----------------START-------------//")
        print("\n")
        print(response)
        print("\n")
        print("//-------------END----------------//")
   
if __name__ == "__main__":
    main()
