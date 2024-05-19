import matplotlib.pyplot as plt
from transformers import AutoTokenizer, DebertaModel
import tiktoken
import torch
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import nest_asyncio # only for jupyter notebook
from transformers import GPT2Tokenizer
from langchain.schema import Document
from typing import Dict, List, Optional, Union, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import os
from getpass import getpass
from semantic_router.encoders import OpenAIEncoder
from semantic_router.splitters import RollingWindowSplitter
from semantic_router.schema import DocumentSplit
from semantic_router.utils.logger import logger
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from semantic_router.encoders import HFEndpointEncoder
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import pyarrow
import uuid
import numpy as np
import torch.nn as nn
import json
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

### env keys
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
load_dotenv()

if __name__ =='__main__':
    print('hola')