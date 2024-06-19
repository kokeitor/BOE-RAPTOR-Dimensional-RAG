import os
import logging
from termcolor import colored
from dotenv import load_dotenv
from VectorDB.db import get_chromadb_retriever, get_pinecone_retriever
from src.GRAPH_RAG.graph import GRAPH_RAG, compile_workflow
from src.GRAPH_RAG.config import ConfigGraph
from src.GRAPH_RAG.graph_utils import (
                        setup_logging,
                        get_arg_parser
                        )


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


# Logging configuration
logger = logging.getLogger(__name__)

## RETRIEVER FUNCTION 
chroma_vectorstore,retriever_chroma,pinecone_vectorstore,retriever_pinecone = db_conexion()
def docs_from_retriver(question :str):
    
    try: 
        return retriever_chroma.invoke(question)
    except Exception as e:
        print(f"{e}")

    try: 
        return retriever_pinecone.invoke(question)
    except Exception as e:
        print(f"{e}")


def main() -> None:
    # Logger set up
    setup_logging()
    retriever, client = get_chromadb_retriever(
    index_name = "hola"
    
                        )
    retriever, client = get_pinecone_retriever(
    index_name = "llama3"
    
            )
    retriever, client = get_pinecone_retriever(
    index_name = "hola"
    
    )
    
    """
     # With scripts parameters mode
    parser = get_arg_parser()
    args = parser.parse_args()
    CONFIG_PATH = args.config_path
    OPENAI_API_KEY = args.token
    DATA_PATH = args.data_path
    MODE = args.mode
    
    # Without scripts parameters mode
    if OPENAI_API_KEY:
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    else:
        os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    if not CONFIG_PATH:
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config/graph', 'graph.json') 
    if not DATA_PATH:
        DATA_PATH = os.path.join(os.path.dirname(__file__), 'config/graph', 'querys.json') 
    if not MODE:
        MODE = 'graph'
        
    logger.info(f"{DATA_PATH=}")
    logger.info(f"{CONFIG_PATH=}")
    logger.info(f"{MODE=}")
    
    # Mode -> Langgraph Agents
    if MODE == 'graph':
        
        logger.info(f"Graph mode")
        logger.info(f"Getting Data and Graph configuration from {DATA_PATH=} and {CONFIG_PATH=} ")
        config_graph = ConfigGraph(config_path=CONFIG_PATH, data_path=DATA_PATH)
        
        logger.info("Creating graph and compiling workflow...")
        graph = create_graph(config=config_graph)
        workflow = compile_workflow(graph)
        logger.info("Graph and workflow created")
        
        thread = {"configurable": {"thread_id": config_graph.thread_id}}
        iteraciones = {"recursion_limit": config_graph.iteraciones}
        
        # itera por todos questions definidos
        for user_question in config_graph.user_questions:
            
            logger.info(f"User Question: {user_question}")
            inputs = {"question": f"{user_question}"}
            
            for event in workflow.stream(inputs, iteraciones):
                for key, value in event.items():
                    print(f"Finished running: {key}:")
            print("BOE DICE : " , value["generation"])
    """

          
if __name__ == '__main__':
    main()
    # terminal command with script parameters : python app.py --data_path ./config/data.json --mode "graph" --config_path ./config/generation.json
    # terminal command : python app.py 



    