import os
import logging
from termcolor import colored
from dotenv import load_dotenv
from langchain.schema import Document
from VectorDB.db import get_chromadb_retriever, get_pinecone_retriever, get_qdrant_retriever
from GRAPH_RAG.graph import create_graph, compile_graph, save_graph
from GRAPH_RAG.config import ConfigGraph
from GRAPH_RAG.models import get_groq
from RAG_EVAL.base_models import RagasDataset 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser, BaseTransformOutputParser

from langchain_core.runnables.config import RunnableConfig

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
    # os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
    os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')
    os.environ['LLAMA_CLOUD_API_KEY'] = os.getenv('LLAMA_CLOUD_API_KEY')
    os.environ['HF_TOKEN'] = os.getenv('HUG_API_KEY')
    os.environ['PINECONE_INDEX_NAME'] = os.getenv('PINECONE_INDEX_NAME')
    os.environ['CHROMA_COLLECTION_NAME'] = os.getenv('CHROMA_COLLECTION_NAME')
    os.environ['QDRANT_API_KEY'] = os.getenv('QDRANT_API_KEY')
    os.environ['QDRANT_HOST'] = os.getenv('QDRANT_HOST')
    os.environ['QDRANT_COLLECTION_NAME'] = os.getenv('QDRANT_COLLECTION_NAME')
    os.environ['QDRANT_COLLECTIONS'] = os.getenv('QDRANT_COLLECTIONS')
    os.environ['APP_MODE'] = os.getenv('APP_MODE')
    os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')
    os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
    
    # Try groq model

    #llm = get_groq()
    #generate_groq_prompt = ChatPromptTemplate.from_messages(
    #        [
    #            (
    #                "system",
    #                """You are an assistant for question-answering tasks.\n
    #                    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.\n
    #                    Use three sentences maximum and keep the answer concise. Provide the answer to the question as a JSON with a single key 'answer'.\n
    #                    Context:\n{context}\n""",
    #            ),
    #            ("human", "{question}"),
    #        ]
    #    )
    #
    #chain = generate_groq_prompt | llm | JsonOutputParser()
    #print(chain.invoke(
    #    {
    #        "question": "¿Cuantos años tienes?",
    #        "context": "Mi nombre es Pedrito"
    #    }
    #))

    # Logger set up
    setup_logging()
    
     # With scripts parameters mode
    parser = get_arg_parser()
    args = parser.parse_args()
    CONFIG_PATH = args.config_path
    DATA_PATH = args.data_path
    MODE = args.mode
    
    # With ENV VARS 
    if not CONFIG_PATH:
        CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..','config/graph', 'graph.json') 
    if not DATA_PATH:
        DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'config/graph', 'querys.json') 
    if not MODE:
        MODE = os.getenv('APP_MODE')
        
    logger.info(f"{DATA_PATH=}")
    logger.info(f"{CONFIG_PATH=}")
    logger.info(f"{MODE=}")
    
    # Mode -> Langgraph Agents
    if MODE == 'graph':
        
        logger.info(f"Graph mode")
        logger.info(f"Getting Data and Graph configuration from {DATA_PATH=} and {CONFIG_PATH=} ")
        config_graph = ConfigGraph(config_path=CONFIG_PATH, data_path=DATA_PATH)
        
        logger.info("Creating graph and compiling workflow...")
        config_graph.graph = create_graph(config=config_graph)
        config_graph.compile_graph = compile_graph(config_graph.graph)
        save_graph(compile_graph=config_graph.compile_graph)
        logger.info("Graph and workflow created")
        
        # RunnableConfig
        runnable_config = RunnableConfig(recursion_limit=config_graph.iteraciones, configurable={"thread_id":config_graph.thread_id})
        
        # itera por todos questions definidos
        for question in config_graph.user_questions:
            
            logger.info(f"User Question: {question.user_question}")
            logger.info(f"User id question: {question.id}")
            inputs = {"question": [f"{question.user_question}"], "date" : question.date}
            
            for event in config_graph.compile_graph.stream(input=inputs,config=runnable_config):
                for key , value in event.items():
                    logger.info(f"Graph event {key} - {value}")
                
if __name__ == '__main__':
    main()




    