import asyncio
import logging
import os

import qdrant_client  # open source vector database
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.legacy.vector_stores import QdrantVectorStore
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_environment():
    load_dotenv()
    logger.info("Environment variables loaded from .env file.")
    data_path = os.getenv('DATA_PATH')

    if data_path is None:
        logger.error("DATA_PATH environment variable is not set")
        raise ValueError("DATA_PATH environment variable is not set")

    logger.info(f"Using DATA_PATH: {data_path}")
    return data_path


def load_documents(data_path):
    logger.info("Loading documents from the specified directory.")
    documents = SimpleDirectoryReader(data_path).load_data()
    logger.info(f"Loaded {len(documents)} documents.")
    return documents


def configure_text_splitter():

    # TextSplitter is used to break down large text documents into smaller chunks.
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)
    logger.info("Text splitter configured with chunk_size=512 and chunk_overlap=10.")
    return text_splitter


def setup_cache():

    # Redis cache speeds up the ingestion process by caching intermediate results.
    redis_cache = RedisCache.from_host_and_port(host="127.0.0.1", port=6379)
    ingest_cache = IngestionCache(
        cache=redis_cache,
        collection="my_test_cache"
    )
    logger.info("Redis cache configured with host=127.0.0.1 and port=6379.")
    return ingest_cache


def setup_vector_store():

    # Qdrant vector store handles efficient storage and retrieval of document vectors.
    client = qdrant_client.QdrantClient(location=":memory:")
    vector_store = QdrantVectorStore(client=client, collection_name="test_store")
    return vector_store


async def run_pipeline(pipeline, documents):
    logger.info("Running ingestion pipeline with async support.")
    nodes = await pipeline.arun(documents=documents)
    logger.info(f"Ingested {len(nodes)} nodes into the vector store.")


def run_pipeline_sync(pipeline, documents):
    logger.info("Running ingestion pipeline with document management.")
    nodes = pipeline.run(documents=documents)
    logger.info(f"Ingested {len(nodes)} nodes with document management.")
    return nodes


def run_pipeline_parallel(pipeline, documents, num_workers):
    logger.info("Running ingestion pipeline with parallel processing.")
    pipeline.run(documents=documents, num_workers=num_workers)
    logger.info("Pipeline execution with parallel processing completed.")


def main():
    data_path = setup_environment()
    documents = load_documents(data_path)
    text_splitter = configure_text_splitter()
    ingest_cache = setup_cache()
    vector_store = setup_vector_store()

    # Configure IngestionPipeline with transformations and storage options.
    pipeline = IngestionPipeline(
        transformations=[
            text_splitter,
            TitleExtractor(),
            TokenTextSplitter(),
        ],
        vector_store=vector_store,
        cache=ingest_cache
    )

    # Run the pipeline asynchronously
    asyncio.run(run_pipeline(pipeline, documents))


if __name__ == "__main__":
    main()
