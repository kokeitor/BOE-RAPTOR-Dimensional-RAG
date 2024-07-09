import os
import requests
import json
from requests import Response
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Union, Optional, Callable, ClassVar
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import logging
import logging.config
import logging.handlers
from ETL.utils import get_current_spanish_date_iso, setup_logging
from ETL.download import WebDownloadData, Downloader

# Logging configuration
logger = logging.getLogger("Download_web_files_module")  # Child logger [for this module]
# LOG_FILE = os.path.join(os.path.abspath("../../../logs/download"), "download.log")  # If not using json config


def main() -> None:
    
    load_dotenv()
    
    # set up the root logger configuration
    setup_logging()
    
    # BOE DOWNLOAD DATA
    BOE_WEB_URL = str(os.getenv('BOE_WEB_URL'))
    BOE_SAVE_PATH = os.path.abspath("./data")
    print(BOE_WEB_URL)
    logger.info(BOE_WEB_URL)
    print(BOE_SAVE_PATH)
    logger.info(BOE_SAVE_PATH)

    data = WebDownloadData(
        web_url=BOE_WEB_URL,
        local_dir=BOE_SAVE_PATH,
        fecha_desde='2024-04-15',
        fecha_hasta='2024-04-15',
        batch=210
    )
    print(data.model_dump())
    logger.info(f"Download information: {data.model_dump()}")
    
    downloader = Downloader(information=data)
    downloader.download()


if __name__ == "__main__":
    main()
