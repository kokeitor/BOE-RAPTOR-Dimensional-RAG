import os
import pandas as pd
from typing import List, Optional
from sklearn.model_selection import train_test_split
from datetime import datetime
from pydantic import BaseModel, Field
from RAPTOR.exceptions import DirectoryNotFoundError
from RAPTOR.utils import get_current_spanish_date_iso
import logging
import logging.handlers
from datasets import DatasetDict
from datasets import Dataset as ds

# Logging configuration
logger = logging.getLogger(__name__)


class HGDataset(BaseModel):
    """
    Class to handle operations related to the Hugging Face Dataset.

    Attributes:
    -----------
    data_dir_path : str
        Directory where .CSV or .parquet files are located.
    hg_api_token : str
        HG Token API key.
    repo_id : str
        HG repository id.
    from_date : str
        Start date of the file name to push to the HG hub.
    to_date : str
        End date of the file name to push to the HG hub.
    desire_columns : Optional[List[str]]
        Columns to get and not drop from data.
    data : Optional[pd.DataFrame]
        Data attribute to store combined data from files.
    """
    data_dir_path: str = Field(default="./", description="Directory where .CSV or .parquet files are")
    hg_api_token: str = Field(description="HG Token API key")
    repo_id: str = Field(description="HG repo id", examples=["<user>/<dataset_name>", "<org>/<dataset_name>"])
    from_date: str = Field(description="First date of the file name to push to the HG hub", examples=["2024-07-12"])
    to_date: str = Field(description="Last date of the file name to push to the HG hub", examples=["2024-07-12"])
    desire_columns: Optional[List[str]] = Field(default=None, description="Columns to get and not drop from data")

    data: Optional[pd.DataFrame] = None  # Define the data attribute

    class Config:
        arbitrary_types_allowed = True

    def initialize_data(self):
        """Initializes the data attribute by cleaning and combining data from files."""
        self.data = self._clean_data(self._get_data())

    def _get_data(self) -> pd.DataFrame:
        """
        Reads and combines data from .CSV and .parquet files within the specified date range.

        Returns:
        --------
        pd.DataFrame
            Combined DataFrame from all the files.
        """
        if not os.path.isdir(self.data_dir_path):
            raise DirectoryNotFoundError(f"The specified directory '{self.data_dir_path}' does not exist.")

        dataframes = []
        for filename in os.listdir(self.data_dir_path):
            logger.info(f"filename : {filename}")
            if "_" in filename and filename[0].isdigit():
                try:
                    file_date = datetime.strptime(filename.split("_")[0], '%Y%m%d%H%M%S')
                except ValueError as e:
                    logger.error(f"Error parsing the date: {e}")
                    continue

                logger.info(f"file_date parsed to correct format: {file_date}")

                if self.parse_date(self.from_date) <= file_date <= self.parse_date(self.to_date):
                    logger.info(f"File name date {file_date} between : {self.parse_date(self.from_date)} and {self.parse_date(self.to_date)}")
                    logger.info(f"Trying to append it")
                    file_path = os.path.join(self.data_dir_path, filename)
                    if filename.endswith('.csv'):
                        df = pd.read_csv(file_path)
                        logger.info(f"Reading CSV file : {file_path}")
                        dataframes.append(df)
                    elif filename.endswith('.parquet'):
                        df = pd.read_parquet(file_path)
                        logger.info(f"Reading parquet file : {file_path}")
                        dataframes.append(df)

        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
        else:
            combined_df = pd.DataFrame()

        return combined_df

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the data by keeping only the desired columns.

        Parameters:
        -----------
        data : pd.DataFrame
            The DataFrame to be cleaned.

        Returns:
        --------
        pd.DataFrame
            The cleaned DataFrame.
        """
        columns_to_keep = []
        if self.desire_columns:
            logger.info(f"Data columns : {data.columns.to_list()}")
            for col in self.desire_columns:
                if col in data.columns.to_list():
                    columns_to_keep.append(col)
                    logger.info(f"Data column to keep {col} exists in file columns")
                else:
                    logger.warning(f"Data column to keep {col} NOT IN file columns")
            return data[columns_to_keep]
        else:
            return data

    @staticmethod
    def parse_date(date_str: str) -> datetime:
        """
        Parses a date string into a datetime object.

        Parameters:
        -----------
        date_str : str
            The date string to be parsed.

        Returns:
        --------
        datetime
            The parsed datetime object.
        """
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            return date_obj
        except ValueError as e:
            raise ValueError(f"Error parsing the date: {e}")

    def _get_hg_dataset(self) -> DatasetDict:
        """
        Splits the data into train, validation, and test sets and removes unnecessary columns.

        Returns:
        --------
        DatasetDict
            The DatasetDict containing train, validation, and test sets.
        """
        try:
            df_train_val, df_test = train_test_split(self.data, test_size=0.2, random_state=42)
            logger.info("Test df shape: %s", df_test.shape)
            df_train, df_val = train_test_split(df_train_val, test_size=0.1, random_state=42)
            logger.info("Train df shape: %s", df_train.shape)
            logger.info("Validation df shape: %s", df_val.shape)
            dataset = DatasetDict({
                "train": ds.from_pandas(df_train, preserve_index=False),
                "validation": ds.from_pandas(df_val, preserve_index=False),
                "test": ds.from_pandas(df_test, preserve_index=False)
            })

            logger.info("\nHG DATASET: %s", dataset)
        except Exception as e:
            logger.error("Error creating HG dataset: %s", e)
            dataset = DatasetDict()

        try:
            # Remove columns from each dataset
            dataset = DatasetDict({
                split: dataset[split].remove_columns(['__index_level_0__'])
                for split in dataset
            })
        except Exception as e:
            logger.error("Error cleaning dataset columns: %s", e)
        return dataset

    def push_to_hub(self, private: Optional[bool] = False, token: Optional[str] = None, branch: Optional[str] = None):
        """
        Pushes the dataset to the Hugging Face Hub.

        Parameters:
        -----------
        private : Optional[bool]
            Whether the repository should be private.
        token : Optional[str]
            The authentication token for the Hugging Face Hub.
        branch : Optional[str]
            The git branch to push the dataset to.
        """
        dataset = self._get_hg_dataset()

        if token is None:
            token = self.hg_api_token

        dataset.push_to_hub(
            repo_id=self.repo_id,
            config_name=get_current_spanish_date_iso(),
            commit_message=f"Date of push: {get_current_spanish_date_iso()}",
            private=private,
            token=token)