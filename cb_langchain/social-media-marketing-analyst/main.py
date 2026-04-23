from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from IPython.display import Markdown, display



def product_description_template() -> str:
    return """
Act as a marketing manager.
Your task is to help a marketing team create a description for a retail website advert of a product based on the 
the technical fact sheet specifications of the product.

Write a brief product description for the website advert.

Technical Specifications:
{fact_sheet_mobile}
"""
def detailed_product_description_template() -> str:
    return """
Your task is to help a marketing team create a description for a retail website advert of a product based on the 
the technical fact sheet specifications of the product.

{product_description_content}

Technical Specifications:
{fact_sheet_mobile}
"""


def invoke_chain (template)-> str:
    chat_template = ChatPromptTemplate.from_template(template)
    chain = chat_template | ChatOllama(model="gemma4:latest")
    response = chain.invoke(
        {
            "fact_sheet_mobile": open("product_factsheet.txt", "r").read(),
            "product_description_content": open("product_description.txt", "r").read()
        }
    )
    return response.content


def main():
    print("Hello from social-media-marketing-analyst!")
    # print(detailed_product_description_template())
    invoke_chain_sequence = [product_description_template(), detailed_product_description_template()]
    for invoke_chain_seq in invoke_chain_sequence:
        response = invoke_chain(invoke_chain_seq)
        # display(Markdown(response.content)) -> Jupyter notebook markdown rendering
        print("//----------------START-------------//")
        print("\n")
        print(response)
        print("\n")
        print("//-------------END----------------//")


if __name__ == "__main__":
    main()
