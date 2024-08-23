import os
import logging
from RAPTOR.exceptions import DirectoryNotFoundError
from RAPTOR.utils import setup_logging
from RAPTOR.RAPTOR_BOE import RaptorDataset
from dotenv import load_dotenv


# Logging configuration
logger = logging.getLogger(__name__)


def main() -> None:
    
    # set up the root logger configuration
    setup_logging()
    
    # Load environment variables from .env file
    load_dotenv()

    os.environ['HG_API_KEY'] = str(os.getenv('HG_API_KEY'))
    
    raptor_dataset = RaptorDataset(
        data_dir_path="./data/boedataset", 
        from_date="2024-07-11", 
        to_date="2024-07-16",
        desire_columns=None # Means all columns
    )
    
    raptor_dataset.initialize_data()
    logger.debug(f"Dataset RAPTOR :  {raptor_dataset.data.head()}")
    logger.debug(f"Dataset RAPTOR columns:  {raptor_dataset.data.columns()}")
    
if __name__ == "__main__":
    main()