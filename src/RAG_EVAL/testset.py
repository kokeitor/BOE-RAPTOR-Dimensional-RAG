from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
import os
import logging
import logging.handlers
from RAPTOR.exceptions import DirectoryNotFoundError
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
import datetime
from dotenv import load_dotenv

# Logging configuration
logger = logging.getLogger(__name__)


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


def get_data(docs_path : str, from_date : str, to_date : str) -> pd.DataFrame:
        """
        Reads and combines data from .CSV and .parquet files within the specified date range.

        Returns:
        --------
        pd.DataFrame
            Combined DataFrame from all the files.
        """
        if not os.path.isdir(docs_path):
            raise DirectoryNotFoundError(f"The specified directory '{docs_path}' does not exist.")

        dataframes = []
        for filename in os.listdir(docs_path):
            logger.info(f"filename : {filename}")
            if "_" in filename and filename[0].isdigit():
                try:
                    file_date = datetime.strptime(filename.split("_")[0], '%Y%m%d%H%M%S')
                except ValueError as e:
                    logger.error(f"Error parsing the date: {e}")
                    continue

                logger.info(f"file_date parsed to correct format: {file_date}")

                if parse_date(from_date) <= file_date <= parse_date(to_date):
                    logger.info(f"File name date {file_date} between : {parse_date(from_date)} and {parse_date(to_date)}")
                    logger.info(f"Trying to append it")
                    file_path = os.path.join(docs_path, filename)
                    if filename.endswith('.csv'):
                        df = pd.read_csv(file_path)
                        logger.info(f"Reading CSV file : {file_path}")
                        logger.info(f"Dataframe columns : {df.columns}")
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


def generate_testset(
                    docs_path : str,
                    from_date : str,
                    to_date : str,
                    generator_llm : ChatOpenAI ,
                    critic_llm : ChatOpenAI , 
                    embedding_model : OpenAIEmbeddings
                    ):
    """_summary_

    Args:
        docs_path (str): _description_
        from_date (str): "2024-08-25"
        to_date (str): "2024-08-25"
        generator_llm (ChatOpenAI, optional): _description_. Defaults to ChatOpenAI(model="gpt-4o-mini").
        critic_llm (ChatOpenAI, optional): _description_. Defaults to ChatOpenAI(model="gpt-4o-mini").
        embedding_model (OpenAIEmbeddings, optional): _description_. Defaults to OpenAIEmbeddings().

    Returns:
        _type_: _description_
    """
    load_dotenv()
    generator_llm = ChatOpenAI(
                        model="gpt-4o-mini",
                        api_key=os.getenv('OPENAI_API_KEY')
                        )
    critic_llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    api_key=os.getenv('OPENAI_API_KEY')
                    )
    
    embedding_model = OpenAIEmbeddings()
    
    generator = TestsetGenerator.from_langchain(
        generator_llm=generator_llm,
        critic_llm=critic_llm,
        embeddings=embedding_model
    )
    
    docs_df = get_data(docs_path=docs_path,from_date=from_date,to_date=to_date) # dataframe with docs
    df_loader = DataFrameLoader(data_frame=docs_df, page_content_column="text")
    docs= df_loader.load() # list of docs
    
    # generate testset
    testset = generator.generate_with_langchain_docs(docs, test_size=10, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})
    
    return testset
