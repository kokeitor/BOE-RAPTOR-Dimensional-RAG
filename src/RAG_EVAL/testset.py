import os
import logging
import pandas as pd
from datetime import datetime
from ragas.testset.generator import TestsetGenerator, TestDataset
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from RAPTOR.exceptions import DirectoryNotFoundError
from ragas.run_config import RunConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from RAG_EVAL.utils import get_current_spanish_date_iso


# Logging configuration
logger = logging.getLogger(__name__)

def parse_date(date_str: str) -> datetime:
    """
    Parses a date string into a datetime object.
    """
    try:
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj
    except ValueError as e:
        raise ValueError(f"Error parsing the date: {e}")

def get_data(docs_path: str, from_date: str, to_date: str) -> pd.DataFrame:
    """
    Reads and combines data from .CSV and .parquet files within the specified date range.
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

def generate_testset(docs_path: str, from_date: str, to_date: str, save_path : str = "./data/rag evaluation/ragas testset") -> TestDataset:
    """
    Generates a test dataset from documents within a specified date range asynchronously.
    """
    generator_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0
    )
    critic_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0
    )
    
    embedding_model = OpenAIEmbeddings()
    
    generator = TestsetGenerator.from_langchain(
        generator_llm=generator_llm,
        critic_llm=critic_llm,
        embeddings=embedding_model
    )
    
    docs_df = get_data(docs_path=docs_path, from_date=from_date, to_date=to_date)  # dataframe with docs
    if docs_df.empty:
        logger.warning("No documents found in the specified date range.")
        return None

    df_loader = DataFrameLoader(data_frame=docs_df, page_content_column="text")
    docs = df_loader.load()  # list of docs
    try:
        logger.info(f"Number of docs to create RAG testset: {len(docs)}")
    except Exception as e:
        logger.error(f"Error logging document count: {e}")
    
    # Add filename key to metadata docs
    if docs:
        for d in docs:
            d.metadata['filename'] = d.metadata.get('pdf_id', 'unknown')
        logger.info(f"Metadata example: {docs[0].metadata}")
    
    # generate testset
    try:
        testset = generator.generate_with_langchain_docs(
            docs, 
            test_size=10, 
            distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
            run_config=RunConfig(max_workers=64)
        )

    except Exception as e:
        logger.error(f"Failed to generate testset: {e}")
        testset = None
        
    # translate question and answer 
    # initialize translator 
    translator = ChatGroq(
                            model= "llama3-70b-8192",
                            temperature=0.0,
                            max_tokens=None,
                            timeout=None,
                            max_retries=10
                                )
    transalate_promt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a assistant tasked with accurately translating sentences from English to Spanish,\n 
                ensuring that the meaning, tone, and context of the original sentence are preserved."""
            ),
            ("human", "{sentence}"),
        ]
    )
    transalate_chain = transalate_promt | translator | StrOutputParser()
    
    # translate 
    if testset:
        testset_df = testset.to_pandas()
        print(testset_df.head())
        print(f"Columnas dataframe : {testset_df.columns}")
        logger.info(f"Columnas dataframe : {testset_df.columns}")
        logger.info(f"dataframe sample : {testset_df.head()}")
        logger.info(f"dataframe len : {testset_df.shape}")
        
        for index, row in enumerate(testset_df.iterrows()):
            try:
                translation = transalate_chain.invoke({"sentence":row["question"]})
                logger.info(f"llm translation of {row['question']}:\n{translation=}")  
                testset_df.loc[index,"question"] = translation
                testset_df.loc[index,"original_question"] = row["question"]
            except Exception as e:
                logger.error(f"llm translation error {e}")  
            
        
    # save testset as csv from pandas dataframe
    if testset:
        # Ensure the path exists; if not, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            logger.info(f"Directory {save_path} created.")

        # Define the full path where the file will be saved
        full_path = os.path.join(save_path, f"{get_current_spanish_date_iso()}_ragas_testset")

        # Save the DataFrame to a CSV file
        testset_df.to_csv(full_path, index=False)
        logger.info(f"DataFrame saved to {full_path}")
        
    return testset

def translate(sentence : str) -> str:
    translator = ChatGroq(
                            model= "llama3-70b-8192",
                            temperature=0.0,
                            max_tokens=None,
                            timeout=None,
                            max_retries=10
                                )
    transalate_promt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a assistant tasked with accurately translating sentences from English to Spanish,\n 
                ensuring that the meaning, tone, and context of the original sentence are preserved."""
            ),
            ("human", "{sentence}"),
        ]
    )
    transalate_chain = transalate_promt | translator | StrOutputParser()
    translation = transalate_chain.invoke({"sentence":sentence})
    logger.info(f"llm translation of {sentence}:\n{translation=}")
    return translation