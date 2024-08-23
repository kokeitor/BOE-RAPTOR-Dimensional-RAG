import os
import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from RAPTOR.exceptions import DirectoryNotFoundError
from RAPTOR.utils import get_current_spanish_date_iso
from ETL.llm import LabelGenerator
import logging
import logging.handlers
import tiktoken
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from typing import Union, Optional
from langchain_nvidia_ai_endpoints import ChatNVIDIA


# Logging configuration
logger = logging.getLogger(__name__)

class ClusterSummaryGenerator:
    def __init__(self, model: str = 'GPT'):
        self.model_label = model
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5")
        
        self.prompt = PromptTemplate(
            template="""You are an assistant specializing in summarizing a text from the Spanish Boletín Oficial del Estado (BOE).
            Your task is to create a summary of the provided text. Be precise and try to collect information from as much of the text as possible.
            Text: {text}""",
            input_variables=["text"],
            input_types={"text":str}
        )
        models = {
            'GPT': ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0),
            'NVIDIA-LLAMA3': ChatNVIDIA(model_name='meta/llama3-70b-instruct', temperature=0),
            'LLAMA': ChatOllama(model='llama3', format="json", temperature=0),
            'LLAMA-GRADIENT': ChatOllama(model='llama3-gradient', format="json", temperature=0)
        }

        self.model = models.get(self.model_label, None)
        if not self.model:
            logger.error("Model ClusterSummaryGenerator Name not correct")
            raise AttributeError("Model ClusterSummaryGenerator Name not correct")

        if self.model_label == "NVIDIA-LLAMA3":
            self.chain = self.prompt | self.model | JsonOutputParser()
        elif self.model_label == "GPT":
            self.chain = self.prompt | self.model | JsonOutputParser()
        else:
            self.chain = self.prompt | self.model | JsonOutputParser()

    def _get_tokens(self, text: str) -> int:
        """Returns the number of tokens in a text string."""
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception as e:
            logger.exception(f"Tokenization error: {e}")
            return len(self.tokenizer(text)["input_ids"])

    def invoke(self, cluster_text : str) -> list[Document]:
        
        cluster_tokens = self._get_tokens(text=cluster_text)

        # Update metadata
        logger.info(f'numero tokens del cluster text : {cluster_tokens}')
        logger.info(f'numero caracteres del cluster text : {len(cluster_text)}')

        try:
            cluster_summary = self.chain.invoke({"text": cluster_text})
            logger.info(f"LLM output: {cluster_summary}")
        except Exception as e:
            logger.exception(f"LLM Error generation cluster summary of {cluster_text} , \nerror message: {e}")
            cluster_summary = "Error in generation of the cluster summary"
                
        return cluster_summary



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
    cluster_summary_generator : ClusterSummaryGenerator = ClusterSummaryGenerator()
    
    class Config:
        arbitrary_types_allowed = True

    def initialize_data(self):
        """Initializes the data attribute by cleaning and combining data from files."""
        self.data = self._clean_data(self._get_data())
        self.data["label_str"] = "" # empty column to fill it with label str 
        self._put_metadata()
        logger.info(f"Dataset RAPTOR sample:\n{self.data.head(1)}")
        logger.info(f"Dataset RAPTOR columns:\n{self.data.columns.to_list()}")
        self.data["cluster_summary"] = "" # empty column to fill it with label str
        self._get_cluster_summary()
        
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
    
    def _put_metadata(self) -> None:
        logger.info(f"Initial labels:\n {LabelGenerator.LABELS}")
        
        # Clean and split the labels
        labels = LabelGenerator.LABELS.replace("\n", "").split(',')
        labels = [label.strip() for label in labels]
        logger.info(f"Processed labels:\n {labels}")
        
        # Create mapping dictionaries
        label2id = {label: label_index for label_index, label in enumerate(labels)}
        id2label = {label_index: label for label_index, label in enumerate(labels)}
        logger.info(f"label2id mapping:\n {label2id}")
        logger.info(f"id2label mapping:\n {id2label}")
        
        # Process each row in the DataFrame
        for index, row in self.data.iterrows():
            if pd.notna(row["label"]):  # Corrected the check for NaN
                
                logger.debug(f"Processing row index: {index}")
                logger.info(f"Row columns: {row.keys()}")
                logger.debug(f"Labels of row {index}: {row['label']}")
                logger.debug(f"Type of object label: {type(row['label'])}")
                
                # Map label ids to label names and add them as a new column to the DataFrame
                labels_id_int = self._parse_label_id_str(row["label"])
                label_columns = [id2label.get(id, "NotExist") for id in labels_id_int]
                logger.info(f"Mapped label columns: {label_columns}")
                
                # Add the label string to the DataFrame
                self.data.at[index, "label_str"] = str(label_columns[0]) if label_columns else "NotExist"
                logger.info(f"Updated self.data.loc[index,'label_str']: {self.data.at[index, 'label_str']}")


    def _parse_label_id_str(self,input_str : str)->None:
        
        str_list = input_str.strip("[]").replace("'", "").split(", ")

        int_list = [int(x) for x in str_list]
        
        logger.info(f"label id input : {input_str}")
        logger.info(f"labels parsed : {int_list}")
        
        return int_list
    
    def _get_cluster_summary(self) -> None:
        unique_labels = self.data["label_str"].unique()
        logger.info(f"unique labels:\n{unique_labels}")
        
        cluster_text = ""
        MAX_LEN = 500 # maximum characters for create cluster summary
        for unique_label in unique_labels:
            filter_dataframe = self.data[self.data["label_str"] == unique_label]
            for text in filter_dataframe["text"]:
                cluster_text = cluster_text + "\n" + str(text)
            logger.info(f"cluster_text for {unique_label=} :\n{cluster_text}")
            summary = self.cluster_summary_generator.invoke(text=cluster_text)
            logger.info(f"cluster summary for {unique_label=} :\n{summary}")
            
        
            

    