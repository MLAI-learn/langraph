from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

llm= ChatGoogleGenerativeAI(model='gemini-2.0-flash')
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | llm | parser

# Run it
result = chain.invoke({"question": "What is the capital of Peru?"})
print(result)
