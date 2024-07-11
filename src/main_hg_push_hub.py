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
            data_dir_path="ruta/al/directorio", 
            hg_api_token="tu_token_aqui", 
            repo_id="tu_repo_id_aqui", 
            from_date="2024-07-12", 
            to_date="2024-07-12",
            desire_columns=["columna1", "columna2"]
        )
        dataset = hg_dataset.get_hg_dataset()
        hg_dataset.push_to_hub(repo_id="tu_repo_id_aqui")
        print("Dataset cargado y subido correctamente.")
    except DirectoryNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Ocurrió un error: {e}")
