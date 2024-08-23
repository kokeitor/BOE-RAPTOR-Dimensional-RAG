import os
import pandas as pd
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from RAPTOR.exceptions import DirectoryNotFoundError
from RAPTOR.utils import get_current_spanish_date_iso
import logging
import logging.handlers


# Logging configuration
logger = logging.getLogger(__name__)


class RaptorDataset(BaseModel):
    """
    Class to handle operations related to the Hugging Face Dataset.

    Attributes:
    -----------
    data_dir_path : str
        Directory where .CSV or .parquet files are located.
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
    from_date: str = Field(description="First date of the file name to push to the HG hub", examples=["2024-07-12"])
    to_date: str = Field(description="Last date of the file name to push to the HG hub", examples=["2024-07-12"])
    desire_columns: Optional[List[str]] = Field(default=None, description="Columns to get and not drop from data")

    data: Optional[pd.DataFrame] = None  # Define the data attribute

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

    