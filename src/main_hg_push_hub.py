import os
import logging
from RAPTOR.exceptions import DirectoryNotFoundError
from RAPTOR.utils import setup_logging
from RAPTOR.hg_push import HGDataset
from dotenv import load_dotenv



# Logging configuration
logger = logging.getLogger(__name__)


def main() -> None:
    
    # set up the root logger configuration
    setup_logging()
    
    # Load environment variables from .env file
    load_dotenv()

    os.environ['HG_API_KEY'] = str(os.getenv('HG_API_KEY'))
    os.environ['HG_REPO_DATASET_ID'] = str(os.getenv('HG_REPO_DATASET_ID'))
    
    hg_dataset = HGDataset(
        data_dir_path="./data/boedataset", 
        hg_api_token=str(os.getenv('HG_API_KEY')), 
        repo_id=str(os.getenv('HG_REPO_DATASET_ID')), 
        from_date="2024-07-10", 
        to_date="2024-07-16",
        desire_columns=["text", "chunk_id","label","pdf_id"]
    )
    
    hg_dataset.initialize_data()
    hg_dataset.push_to_hub()
    print("Dataset cargado y subido correctamente.")
    
# Ejemplo de uso
if __name__ == "__main__":
    main()