import os
import logging
import asyncio
from RAG_EVAL.testset import generate_testset
from RAG_EVAL.utils import setup_logging
from dotenv import load_dotenv

# Logging configuration
logger = logging.getLogger(__name__)


def main() -> None:
    
    # Load environment variables from .env file
    load_dotenv()

    # Set environment variables
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')

    # set up the root logger configuration
    setup_logging()
    
    # Create Raptor data (make cluster summary, process and store in vector database)
    docs_path="./data/boedataset" 
    from_date="2024-08-28"
    to_date="2024-08-30"
    testset = generate_testset(docs_path, from_date, to_date)
    
if __name__ == "__main__":
    main()