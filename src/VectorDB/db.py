from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores.kinetica import DistanceStrategy
import chromadb
import logging 
import os
from dotenv import load_dotenv
from typing import Union
from GRAPH_RAG.models import get_openai_emb, get_hg_emb
from exceptions.exceptions import VectorDatabaseError
import VectorDB.test_db
"""
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
"""


INDEX_NAME = "INDEX_DEFAULT_VDB_NAME"

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Set environment variables
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')


@VectorDB.test_db.try_retriever(query="¿que dia es hoy?")
@VectorDB.test_db.try_client_conexion
def get_chromadb_retriever(
                            index_name :str = INDEX_NAME, 
                            get_embedding_model : callable = get_hg_emb, 
                            embedding_model : str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                            collection_metadata : dict[str,str] = {"hnsw:space": "cosine"},
                            search_kwargs : dict = {"k" : 3},
                            delete_index_name : Union[str,None] = None
                            ) -> tuple[VectorStoreRetriever,VectorStore]:
    try:
        client = chromadb.HttpClient(host='localhost', port=8000)
    except ValueError:
        logger.error("ValueError: Could not connect to a Chroma server. Are you sure it is running?")
    
    if delete_index_name:
        logger.error(f"Try to delete existing collection name index ->  '{delete_index_name}' of client CHROMA DB")
        try : 
            client.delete_collection(name=delete_index_name)
            logger.info(f"CHROMA DB collection with name -> '{delete_index_name}' deleted")
        except Exception as e:
            logger.error(f"No CHROMA DB collection with name : {delete_index_name}")
            raise VectorDatabaseError(message="Error while connecting to Chroma DB",exception=e)
        
    try: 
        chroma_vectorstore = Chroma(
                                    embedding_function=get_embedding_model(model=embedding_model),   
                                    client=client,
                                    collection_name=index_name,
                                    collection_metadata=collection_metadata
                                    )
    except Exception as e:
        logger.error(f"Error while connecting to Chroma DB -> {e}")
        raise VectorDatabaseError(message="Error while connecting to Chroma DB",exception=e)
  
    retriever =  chroma_vectorstore.as_retriever(search_kwargs = search_kwargs)
    
    return retriever , chroma_vectorstore


@VectorDB.test_db.try_retriever(query="¿hola?")
@VectorDB.test_db.try_client_conexion
def get_pinecone_retriever(
                            index_name :str = INDEX_NAME, 
                            get_embedding_model : callable = get_hg_emb, 
                            embedding_model : str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" ,
                            search_kwargs : dict = {"k" : 3},
                            ) -> tuple[VectorStoreRetriever,VectorStore]:

    try:
        logger.info(f"Connecting to an existing index of PineCone DB cient -> {index_name}")
        pinecone_vectorstore = PineconeVectorStore(
                                                embedding=get_embedding_model(model=embedding_model),
                                                text_key='text',
                                                distance_strategy=DistanceStrategy.COSINE,
                                                pinecone_api_key=os.getenv('PINECONE_API_KEY'),
                                                index_name=index_name
                                                )
    except Exception as e:
        logger.error(f"Error while connecting to PineCone DB from existing index : {index_name} -> {e}")
        raise VectorDatabaseError(message="Error while connecting to Chroma DB",exception=e)
        
    retriever = pinecone_vectorstore.as_retriever(search_kwargs = search_kwargs)
    
    return retriever , pinecone_vectorstore


"""
def get_qdrant_retriever(   
                        index_name :str = INDEX_NAME, 
                        embedding_model : callable = get_hg_emb, 
                        search_kwargs : dict = {"k" : 3},
                        new_index_name : Union[str, None] = None
                            ) -> VectorStoreRetriever:
    qdrant_client = QdrantClient(
        url="https://31f87457-62d5-4e83-8645-06687c43390f.europe-west3-0.gcp.cloud.qdrant.io:6333", 
        api_key=os.getenv('QDRANT_API_KEY'),
    )

    logger.info(f"Existing QDRANT collections : {qdrant_client.get_collections()}")
    return None

"""

