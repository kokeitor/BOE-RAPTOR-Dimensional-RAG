import os
import tiktoken
import pytz
from transformers import AutoTokenizer, DebertaModel, GPT2Tokenizer
from dotenv import load_dotenv
from typing import Dict, List, Union, Optional
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Union, Optional, Callable, ClassVar
import logging


# Load environment variables from .env file
load_dotenv()


# Set environment variables
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')
os.environ['LLAMA_CLOUD_API_KEY'] = os.getenv('LLAMA_CLOUD_API_KEY')
os.environ['HF_TOKEN'] = os.getenv('HUG_API_KEY')


# Tokenizers
TOKENIZER_GPT3 = tiktoken.encoding_for_model("gpt-3.5")
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
TOKENIZER_LLAMA3 = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer_deberta = AutoTokenizer.from_pretrained("microsoft/deberta-base")


# Embedding model
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# Logging configuration
logger = logging.getLogger("parser_module_logger")  # Child logger [for this module]
# LOG_FILE = os.path.join(os.path.abspath("../../../logs/download"), "download.log")  # If not using json config


# util functions
def get_current_spanish_date_iso():
    # Get the current date and time in the Europe/Madrid time zone
    spanish_tz = pytz.timezone('Europe/Madrid')
    return datetime.now(spanish_tz).strftime("%Y%m%d%H%M%S")

class Parser:
    def __init__(self, directory_path: str, 
                 file_type: str = ".pdf", 
                 recursive_parser: bool = True, 
                 result_type: str = "markdown", 
                 verbose: bool = True, 
                 api_key: str = os.getenv('LLAMA_CLOUD_API_KEY')
                 ):
        self.path = directory_path
        self.parser = LlamaParse(
                                    api_key=api_key,
                                    result_type=result_type,  # "markdown" and "text" are available
                                    verbose=verbose
                                )

        self.reader = SimpleDirectoryReader(
            input_dir=self.path,
            file_extractor={file_type: self.parser},
            recursive=recursive_parser,  # recursively search in subdirectories
            required_exts=[file_type]
        )
        
    def invoke(self) -> List[Document]:
        
        self.llama_parsed_docs = self.reader.load_data()  # returns List[llama doc objt]
        self.lang_parsed_docs = [d.to_langchain_format() for d in self.llama_parsed_docs]
    
        if len(self.lang_parsed_docs) == 0:
            logger.error("Parsed docs list empty")
        else:
            logger.info(f"Parsed num of docs -> {len(self.lang_parsed_docs) }")
        return self.lang_parsed_docs
