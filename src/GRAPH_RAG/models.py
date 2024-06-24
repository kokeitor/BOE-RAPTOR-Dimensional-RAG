import logging
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from langchain_community.embeddings import GPT4AllEmbeddings 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.chat_models import ChatOllama
import warnings


# Logging configuration
logger = logging.getLogger(__name__)


# Suppress all warnings
warnings.filterwarnings("ignore")


def get_open_ai_json(temperature=0, model='gpt-3.5-turbo'):
    """
    _summary_
    Args:
        temperature (int, optional): _description_. Defaults to 0.
        model (str, optional): _description_. Defaults to 'gpt-3.5-turbo'.
    """
    logger.info(f"Using Open AI : {model}")
    llm = ChatOpenAI(
                    model=model,
                    temperature = temperature,
                    model_kwargs={"response_format": {"type": "json_object"}},
                    )
    return llm


def get_open_ai(temperature=0, model='gpt-3.5-turbo'):
    """
    _summary_
    Args:
        temperature (int, optional): _description_. Defaults to 0.
        model (str, optional): _description_. Defaults to 'gpt-3.5-turbo'.
    """
    logger.info(f"Using Open AI : {model}")
    llm = ChatOpenAI(
                    model=model,
                    temperature = temperature
                    )
    return llm


def get_nvdia(temperature=0, model='meta/llama3-70b-instruct'):
    """
    Nvidia llama 3 model
    Args:
        temperature (int, optional): _description_. Defaults to 0.
        model (str, optional): _description_.
    """
    logger.info(f"Using NVIDIA : {model}")
    llm = ChatNVIDIA(
                    model=model,
                    temperature = temperature
                    )
    return llm


def get_ollama(temperature=0, model='llama3'):
    """
    Ollama local model
    Args:
        temperature (int, optional): _description_. Defaults to 0.
        model (str, optional): _description_.
    """
    logger.info(f"Using Ollama : {model}")
    llm = ChatOllama(
                    model=model,
                    temperature = temperature,
                    format="json"
                    )
    return llm

def get_hg_emb(model : str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """_summary_

    Args:
        model (str): _description_
    """
    logger.info(f"Using HuggingFace embedding model : {model}")
    sbert = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                                )
    return sbert

def get_openai_emb(model : str = "all‑MiniLM‑L6‑v2.gguf2.f16.gguf"):
    """_summary_

    Args:
        model (str): _description_
    """
    logger.info(f"Using HuggingFace embedding model : {model}")
    llm = GPT4AllEmbeddings(
                            model_name ="all‑MiniLM‑L6‑v2.gguf2.f16.gguf"
                            )
    return llm