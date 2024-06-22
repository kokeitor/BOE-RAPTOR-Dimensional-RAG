import os
import logging
from termcolor import colored
from dotenv import load_dotenv
from langchain.schema import Document
from VectorDB.db import get_chromadb_retriever, get_pinecone_retriever, get_qdrant_retriever
from GRAPH_RAG.graph import create_graph, compile_graph, save_graph
from GRAPH_RAG.config import ConfigGraph
from app.app import run_app
from GRAPH_RAG.graph_utils import (
                        setup_logging,
                        get_arg_parser
                        )


# Logging configuration
logger = logging.getLogger(__name__)


def main() -> None:
    
    # Load environment variables from .env file
    load_dotenv()

    # Set environment variables
    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
    os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')
    os.environ['LLAMA_CLOUD_API_KEY'] = os.getenv('LLAMA_CLOUD_API_KEY')
    os.environ['HF_TOKEN'] = os.getenv('HUG_API_KEY')
    os.environ['PINECONE_COLLECTION_NAME'] = os.getenv('PINECONE_COLLECTION_NAME')
    os.environ['CHROMA_COLLECTION_NAME'] = os.getenv('CHROMA_COLLECTION_NAME')
    os.environ['QDRANT_API_KEY'] = os.getenv('QDRANT_API_KEY')
    os.environ['QDRANT_HOST'] = os.getenv('QDRANT_HOST')
    os.environ['QDRANT_COLLECTION_NAME'] = os.getenv('QDRANT_COLLECTION_NAME')
    os.environ['QDRANT_COLLECTIONS'] = os.getenv('QDRANT_COLLECTIONS')
    os.environ['APP_MODE'] = os.getenv('APP_MODE')
    os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')
 
    # Logger set up
    setup_logging()
    
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..','config/graph', 'graph.json') 
    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'config/graph', 'querys.json') 
        
    logger.info(f"{DATA_PATH=}")
    logger.info(f"{CONFIG_PATH=}")
    logger.info(f"Streamlit app mode")
    
    logger.info(f"Getting Data and Graph configuration from {DATA_PATH=} and {CONFIG_PATH=} ")
    config_graph = ConfigGraph(config_path=CONFIG_PATH, data_path=DATA_PATH)
    
    logger.info("Creating graph and compiling workflow...")
    config_graph.graph = create_graph(config=config_graph)
    config_graph.compile_graph = compile_graph(config_graph.graph)
    # save_graph(workflow)
    logger.info("Graph and workflow created")
    
    # Run streamlit app
    run_app(config_graph=config_graph)
    
if __name__ == '__main__':
    
    main()