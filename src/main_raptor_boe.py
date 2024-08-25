import os
import logging
from RAPTOR.exceptions import DirectoryNotFoundError
from RAPTOR.utils import setup_logging
from RAPTOR.RAPTOR_BOE import RaptorDataset
from dotenv import load_dotenv


# Logging configuration
logger = logging.getLogger(__name__)


def main() -> None:
    
    # Load environment variables from .env file
    load_dotenv()

    # Set environment variables
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    os.environ['QDRANT_API_KEY'] = os.getenv('QDRANT_API_KEY')
    os.environ['QDRANT_HOST'] = os.getenv('QDRANT_HOST')
    os.environ['QDRANT_COLLECTION_NAME'] = os.getenv('QDRANT_COLLECTION_NAME')
    os.environ['QDRANT_COLLECTIONS'] = os.getenv('QDRANT_COLLECTIONS')
    os.environ['HG_API_KEY'] = str(os.getenv('HG_API_KEY'))
    os.environ['PINECONE_API_KEY'] = str(os.getenv('PINECONE_API_KEY'))
    os.environ['PINECONE_INDEX_NAME'] = str(os.getenv('PINECONE_INDEX_NAME'))

    # set up the root logger configuration
    setup_logging(script="raptor_boe")
    
    # Create Raptor data (make cluster summary, process and store in vector database)
    raptor_dataset = RaptorDataset(
        data_dir_path="./data/boedataset", 
        from_date="2024-07-11", 
        to_date="2024-07-16",
        desire_columns=None # Means all columns
    )
    raptor_dataset.initialize_data()
    
    # Store in vector database
    
    
    
if __name__ == "__main__":
    main()