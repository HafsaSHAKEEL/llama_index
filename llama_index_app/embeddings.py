import logging
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.huggingface_optimum import OptimumEmbedding


from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from InstructorEmbedding import INSTRUCTOR
from llama_index.core.embeddings import BaseEmbedding
from typing import List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_openai_embedding():
    logger.info("Setting up OpenAI embedding model.")
    embed_model = OpenAIEmbedding()
    Settings.embed_model = embed_model
    return embed_model

def setup_huggingface_embedding(model_name: str):
    logger.info(f"Setting up Hugging Face embedding model with model_name={model_name}.")
    embed_model = HuggingFaceEmbedding(model_name=model_name)
    Settings.embed_model = embed_model
    return embed_model

def setup_optimum_embedding(folder_name: str):
    logger.info(f"Setting up Optimum embedding model with folder_name={folder_name}.")
    embed_model = OptimumEmbedding(folder_name=folder_name)
    Settings.embed_model = embed_model
    return embed_model

def setup_langchain_embedding(model_name: str):
    logger.info(f"Setting up Langchain embedding model with model_name={model_name}.")
    embed_model = HuggingFaceBgeEmbeddings(model_name=model_name)
    Settings.embed_model = embed_model
    return embed_model

class InstructorEmbeddings(BaseEmbedding):
    def __init__(self, instructor_model_name: str = "hkunlp/instructor-large", instruction: str = "Represent the Computer Science documentation or question:", **kwargs: Any) -> None:
        self._model = INSTRUCTOR(instructor_model_name)
        self._instruction = instruction
        super().__init__(**kwargs)

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, query]])
        return embeddings[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, text]])
        return embeddings[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode([[self._instruction, text] for text in texts])
        return embeddings

    async def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

def setup_custom_instructor_embendding(instructor_model_name: str = "hkunlp/instructor-large", instruction: str = "Represent the Computer Science documentation or question:"):
    logger.info(f"Setting up Instructor embedding model with instructor_model_name={instructor_model_name} and instruction={instruction}.")
    embed_model = InstructorEmbeddings(instructor_model_name=instructor_model_name, instruction=instruction)
    Settings.embed_model = embed_model
    return embed_model

def get_text_embedding(embed_model, text: str):

    logger.info(f"Getting text embedding for: {text}")
    embedding = embed_model.get_text_embedding(text)
    return embedding
