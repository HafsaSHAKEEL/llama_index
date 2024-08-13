from llama_index.llms.openai import OpenAI
import openai
from dotenv import load_dotenv
import os


load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")


llama_openai = OpenAI(api_key=openai.api_key)


response = llama_openai.complete("Paul Graham is ")


print(response)
