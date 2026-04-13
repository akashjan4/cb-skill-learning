from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
import reviews

review_data = reviews.product_reviews
prompt_file_name = "prompt.txt"
llm = ChatOllama(model="gemma4:latest")
# Define your desired data structure - like a python data class.


class ReviewAnalysisResponse(BaseModel):
    summary: str = Field(
        description="A concise summary of the review with maximum 300 characters."
    )
    positive: list = Field(
        description="A list of positive aspects mentioned in the review."
    )
    negative: list = Field(
        description="A list of negative aspects mentioned in the review."
    )
    sentiment: str = Field(
        description="Overall sentiment of the review. It can be either Positive, Negative or Neutral."
    )
    emotions: list = Field(
        description="A list of emotions expressed in the review. It can include emotions like happiness, frustration, excitement, disappointment etc."
    )
    email: str = Field(
        description="Details email to the customer based on the sentiment"
    )


parser = PydanticOutputParser(pydantic_object=ReviewAnalysisResponse)

prompt_txt = open(prompt_file_name, "r").read()

prompt = PromptTemplate(
    template=prompt_txt,
    input_variables=["review"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)


def main():
    print("Hello from review-analyst!")
    chain = prompt | llm | parser
    responses = chain.map().invoke(review_data)
    for response in responses:
        for key, value in response.dict().items():
            print(f"{key.upper()}: {value}")
            print("\n")
        print("-------------------------------")
        print("\n\n")


if __name__ == "__main__":
    main()
