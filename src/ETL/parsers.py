import os
from langchain.schema import Document
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import logging


# Logging configuration
logger = logging.getLogger("parser_module_logger")  # Child logger [for this module]
# LOG_FILE = os.path.join(os.path.abspath("../../../logs/download"), "download.log")  # If not using json config

class Parser:
    def __init__(self, 
                 directory_path: str, 
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
        
    def invoke(self) -> list[Document]:
        
        self.llama_parsed_docs = self.reader.load_data()  # returns List[llama doc objt]
        self.lang_parsed_docs = [d.to_langchain_format() for d in self.llama_parsed_docs]
    
        if len(self.lang_parsed_docs) == 0:
            logger.error("Parsed docs list empty")
        else:
            logger.info(f"Parsed num of docs -> {len(self.lang_parsed_docs) }")
        return self.lang_parsed_docs
