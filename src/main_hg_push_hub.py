import os
import logging
from RAPTOR.exceptions import DirectoryNotFoundError
from RAPTOR.utils import setup_logging
from RAPTOR.hg import HGDataset
from dotenv import load_dotenv



# Logging configuration
logger = logging.getLogger(__name__)


def main() -> None:
    
    # set up the root logger configuration
    setup_logging()
    
    # Load environment variables from .env file
    load_dotenv()

    os.environ['HG_API_KEY'] = os.getenv('HG_API_KEY')
    os.environ['HG_REPO_DATASET_ID'] = os.getenv('HG_REPO_DATASET_ID')
        
# Ejemplo de uso
if __name__ == "__main__":
    try:
        hg_dataset = HGDataset(
            data_dir_path="./data/boedataset", 
            hg_api_token=os.getenv('HG_API_KEY'), 
            repo_id=os.getenv('HG_REPO_DATASET_ID'), 
            from_date="2024-04-15", 
            to_date="2024-04-15",
            desire_columns=["text", "chunk_id","label"]
        )
        dataset = hg_dataset.get_hg_dataset()
        hg_dataset.hg_dataset.push_to_hub()
        print("Dataset cargado y subido correctamente.")
    except DirectoryNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Ocurrió un error: {e}")
