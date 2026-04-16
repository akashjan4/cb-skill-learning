# orig_msg: The original customer message
# orig_lang: Detected language of the customer message e.g. Spanish
# category: 1-2 word describing the category of the problem
# trans_msg: Translated customer message in English
# response: Response to the customer in orig_lang
# trans_response: Response to the customer in English

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
import pandas as pd

class ITSupportResponse(BaseModel):
	orig_msg: str = Field(description="The original customer message")
	orig_lang: str = Field(description="Detected language of the customer message e.g. Spanish")
	category: str = Field(description="1-2 word describing the category of the problem")
	trans_msg: str = Field(description="Translated customer message in English")
	response: str = Field(description="Response to the customer in orig_lang")
	trans_response: str = Field(description="Response to the customer in English")


parser = JsonOutputParser(pydantic_object=ITSupportResponse)
prompt_text = """
	Act as an Information Technology (IT) customer support agent.
	For the IT support message mentioned below
	use the following output format when generating the output response

	Output format instructions:
	{format_instructions}

	Customer IT support message:
	{it_support_msg}"""

prompt = PromptTemplate(
	template=prompt_text,
	input_variables=["it_support_msg"],
	partial_variables={"format_instructions": parser.get_format_instructions()}
)

llm = ChatOllama(model="gemma4:latest")
chain = (prompt | llm | parser)

it_support_queue = [
	"Não consigo sincronizar meus contatos com o telefone. Sempre recebo uma mensagem de falha.",
	"Ho problemi a stampare i documenti da remoto. Il lavoro non viene inviato alla stampante di rete.",
	"プリンターのトナーを交換しましたが、印刷品質が低下しています。サポートが必要です。",
	"Я не могу войти в систему учета времени, появляется сообщение об ошибке. Мне нужна помощь.",
	"Internet bağlantım cok yavas ve bazen tamamen kesiliyor. Yardım eder misiniz?",
	"Не могу установить обновление безопасности. Появляется код ошибки. Помогите, пожалуйста."
]

def main():
	print("Hello from it-support-analyst!")
	formatted_msgs = [{"it_support_msg": msg} for msg in it_support_queue]
	responses = chain.map().invoke(formatted_msgs)
	for i, response in enumerate(responses):
		print(f"//----------------START RESPONSE {i+1}-------------//")
		print("\n")
		print(response)
		print("\n")
		print(f"//-------------END RESPONSE {i+1}----------------//")
	# df = pd.DataFrame(responses)
	# print(df)


if __name__ == "__main__":
	main()
