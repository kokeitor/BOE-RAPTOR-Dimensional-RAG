import torch.nn as nn
import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data.dataset import ConcatDataset
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
from typing import List, Tuple, Dict, Optional, Union
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
from transformers.integrations import TensorBoardCallback
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig, EarlyStoppingCallback
from sklearn.metrics import f1_score
from datasets import Dataset
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from RAPTOR.exceptions import DirectoryNotFoundError
import logging
import logging.config
import logging.handlers
from datasets import DatasetDict


# Logging configuration
logger = logging.getLogger(__name__)


# Dataset
class HGDataset(BaseModel):
    data_dir_path: str = Field(default="./", description="Directory where .CSV or .parquet files are")
    hg_api_token: str = Field(description="HG Token API key")
    repo_id: str = Field(description="HG repo id", examples=["<user>/<dataset_name>", "<org>/<dataset_name>"])
    from_date: str = Field(description="First date of the file name to push to the HG hub", examples=["2024-07-12"])
    to_date: str = Field(description="Last date of the file name to push to the HG hub", examples=["2024-07-12"])
    desire_columns: Optional[List[str]] = Field(default=None, description="Columns to get and not drop from data ")
    
    def __post_init__(self):
        self.data = self.clean_data(self.get_data())
        self.hg_dataset = self.get_hg_dataset()
        
    
    def get_data(self) -> pd.DataFrame:
        # Verifica si el directorio existe
        if not os.path.isdir(self.data_dir_path):
            raise DirectoryNotFoundError(f"The specified directory '{self.data_dir_path}' does not exist.")
        
        # Inicializa una lista para almacenar DataFrames
        dataframes = []

        # Itera sobre los archivos en el directorio
        for filename in os.listdir(self.data_dir_path):
            file_date = datetime.strptime(filename.split("_")[0], '%Y%m%d%H%M%S')
            logger.info(f"file_date parsed to correct format: {file_date}")
            
            if self.parse_date(self.from_date) <= file_date <= self.parse_date(self.to_date):
                file_path = os.path.join(self.data_dir_path, filename)

                # Comprueba si el archivo es un .csv o .parquet
                if filename.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    dataframes.append(df)
                elif filename.endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                    dataframes.append(df)
        
        # Apila todos los DataFrames en uno solo
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
        else:
            combined_df = pd.DataFrame()
        
        return combined_df
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.desire_columns: 
            return data[self.desire_columns]
        return data
    

    @staticmethod
    def parse_date(date_str: str) -> datetime:
        """
        Convierte una fecha en formato 'YYYY-MM-DD' a un objeto datetime.

        Parámetros:
        date_str (str): La fecha en formato 'YYYY-MM-DD'.
        
        Retorna:
        datetime: Objeto datetime con la hora, minutos y segundos establecidos a '000000'.
        """
        try:
            # Parsear la fecha de entrada
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            return date_obj
        except ValueError as e:
            raise ValueError(f"Error al parsear la fecha: {e}")

    def get_hg_dataset(self) -> DatasetDict:
        try:
            # test 20 % train
            df_train_val, df_test = train_test_split(self.data, test_size=0.2, random_state=42)
            logger.info("Test df shape: %s", df_test.shape)
            
            # Validation 10% de train
            df_train, df_val = train_test_split(df_train_val, test_size=0.1, random_state=42)
            logger.info("Train df shape: %s", df_train.shape)
            logger.info("Validation df shape: %s", df_val.shape)
            logger.info('---------------------------------------------------\n')
            
            dataset = DatasetDict({
                "train": Dataset.from_pandas(df_train),
                "validation": Dataset.from_pandas(df_val),
                "test": Dataset.from_pandas(df_test)
            })

            logger.info("\nHG DATASET: %s", dataset)
        except Exception as e:
            logger.error("Error creating HG dataset: %s", e)
            dataset = DatasetDict()

        return dataset

    def push_to_hub(self, 
                    private: Optional[bool] = False, 
                    token: Optional[str] = None, 
                    branch: Optional[str] = None, 
                    shard_size: Optional[int] = 524288000):
        """
        Pushes the DatasetDict to the Hugging Face Hub.

        Parameters:
        repo_id (str): The ID of the repository to push to in the following format: <user>/<dataset_name> or <org>/<dataset_name>. Also accepts <dataset_name>, which will default to the namespace of the logged-in user.
        private (Optional bool): Whether the dataset repository should be set to private or not. Only affects repository creation: a repository that already exists will not be affected by that parameter.
        token (Optional str): An optional authentication token for the Hugging Face Hub. If no token is passed, will default to the token saved locally when logging in with huggingface-cli login. Will raise an error if no token is passed and the user is not logged-in.
        branch (Optional str): The git branch on which to push the dataset.
        shard_size (Optional int): The size of the dataset shards to be uploaded to the hub. The dataset will be pushed in files of the size specified here, in bytes.
        """

        if token is None:
            token = self.hg_api_token

        self.hg_dataset.push_to_hub(
            repo_id=self.repo_id,
            private=private,
            token=self.hg_api_token,
            branch=branch,
            shard_size=shard_size
        )
