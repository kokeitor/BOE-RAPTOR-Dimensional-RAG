from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
import chromadb
import logging 
from dataclasses import dataclass
from typing import Union, ClassVar
from GRAPH_RAG.models import get_openai_emb, get_hg_emb
from pydantic import BaseModel
import test_db


INDEX_NAME = "INDEX_DEFAULT_VDB_NAME"

logger = logging.getLogger(__name__)


@test_db.try_client_conexion(tries = 3)
@test_db.try_retriever(try_query="¿que dia es hoy?")
def get_chromadb_retriever(
                            index_name :str = INDEX_NAME, 
                            embedding_model : Embeddings = get_hg_emb, 
                            collection_metadata : dict[str,str] = {"hnsw:space": "cosine"},
                            search_kwargs : dict = {"k" : 3}
                            ) -> tuple[VectorStoreRetriever,]:
    
    client = chromadb.HttpClient(host='localhost', port=8000)
    try: 
        chroma_vectorstore = Chroma(
                                embedding_function=embedding_model,   
                                client=client,
                                collection_name=index_name,
                                collection_metadata=collection_metadata
                                )
    except Exception as e:
        logger.error(f"Error while connecting to Chroma DB -> {e}")
        
    retriever =  chroma_vectorstore.as_retriever(search_kwargs = search_kwargs)
    
    return retriever , chroma_vectorstore

    
@test_db.try_client_conexion(tries = 3)
@test_db.try_retriever(try_query="¿que dia es hoy?")
def get_pinecone_retriever(
                            index_name :str = INDEX_NAME, 
                            embedding_model : Embeddings = get_hg_emb, 
                            search_kwargs : dict = {"k" : 3}
                            ) -> VectorStoreRetriever:
    try:
        pinecone_vectorstore = PineconeVectorStore.from_existing_index(
                                                                        index_name = index_name, 
                                                                        embedding = embedding_model
                                                                    )
    except Exception as e:
        logger.error(f"Error while connecting to PineCone DB -> {e}")
        
    retriever = pinecone_vectorstore.as_retriever(search_kwargs = search_kwargs)
    
    return retriever , pinecone_vectorstore


def get_qdrant_retriever(embedding_model):
    """_summary_

    Args:
        embedding_model (_type_): _description_

    Returns:
        _type_: _description_
    """
    return None

