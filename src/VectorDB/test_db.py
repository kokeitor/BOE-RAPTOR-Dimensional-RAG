from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
import chromadb
import logging 
from dataclasses import dataclass
from typing import Union, ClassVar
from GRAPH_RAG.models import get_openai_emb


logger = logging.getLogger(__name__)



def try_client_conexion(tries: int = 2):
    """
    Wrapper -> try connection to client vector db

    Args:
        tries (int, optional): Number of attempts to try connection. Defaults to 2.
    """
    def decorator(func: callable):
        def wrapper(*args, **kwargs):
            retriever, vectorstore = func(*args, **kwargs)
            logger.info(f"Trying client db {func.__name__}")

            # First test
            try:
                for t in range(tries):
                    for id in vectorstore.get()["ids"]:
                        logger.info(f"Try {t} for id : {id} -> {vectorstore.get(include=['embeddings', 'documents', 'metadatas'])}")
            except Exception as e:
                logger.error(f"Client db First test error using {func.__name__} -> {e}")

            # Second test
            try:
                # Note: Instantiating Chroma class creates an object equivalent to chroma_client from chromadb library but using langchain library
                logger.info("Db Collection info : ", vectorstore.get().keys())
                logger.info("Db Collection info ids len : ", (vectorstore.get()["ids"]))
                logger.info("Db Collection docs : ", vectorstore.get()["documents"])
                try:
                    logger.info(f"Db Collection embeddings (1st comp of first embedding) :  {vectorstore.get(include=['embeddings'])['embeddings'][0][0]}")
                    logger.info(f"Db len of collection embeddings: {len(vectorstore.get(include=['embeddings'])['embeddings'][0])}")
                except Exception as e:
                    logger.error(f"Client db Second test [embeddings] error using {func.__name__} -> {e}")
            except Exception as e:
                logger.error(f"Client db Second test [Collection] error using {func.__name__} -> {e}")

            return retriever, vectorstore

        return wrapper
    return decorator


def try_retriever(query="La duración total de las enseñanzas en ciclos de grado medio"):
    """
    Try retriever

    Args:
        query (str, optional): The query to test the retriever. Defaults to "La duración total de las enseñanzas en ciclos de grado medio".
    """
    def decorator(func: callable):
        def wrapper(*args, **kwargs):
            retriever, vectorestore = func(*args, **kwargs)
            logger.info(f"Trying retriever {func.__name__}")

            try:
                response = retriever.invoke(query)
                logger.info(f"Number of embeddings retrieved : {len(response)}")
                if len(response) > 0:
                    logger.info(f"Best similarity retriever search : {response[0].page_content}")
            except Exception as e:
                logger.error(f"Retriever error using {func.__name__} -> {e}")

            return retriever, vectorestore

        return wrapper
    return decorator

