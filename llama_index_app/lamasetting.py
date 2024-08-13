import os
import openai
import logging
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings  #  custom setup to globally manage configurations.
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
data_path = os.getenv("DATA_PATH")

logger.info(f"Loaded OpenAI API Key: {'******' if openai.api_key else 'Not Found'}")
logger.info(f"Data Path: {data_path}")

Settings.llm = OpenAI(temperature=0.2, model="gpt-4o-mini")
logger.info(f"LLM Settings: Temperature={Settings.llm.temperature}, Model={Settings.llm.model}")

try:
    documents = SimpleDirectoryReader(data_path).load_data()
    logger.info(f"Number of documents loaded: {len(documents)}")
except Exception as e:
    logger.error(f"Failed to load documents: {e}")
    documents = None

if documents:
    try:
        index = VectorStoreIndex.from_documents(documents)
        logger.info("VectorStoreIndex created successfully.")
    except Exception as e:
        logger.error(f"Failed to create VectorStoreIndex: {e}")
else:
    logger.error("No documents available to create VectorStoreIndex.")
