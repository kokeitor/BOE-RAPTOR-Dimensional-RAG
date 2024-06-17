import logging
import logging.config
import logging.handlers
from typing import List, Dict
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from ..exceptions.exceptions import LangChainError
from prompts import (
    analyze_cv_prompt,
    offer_check_prompt,
    re_analyze_cv_prompt,
    cv_check_prompt,
    analyze_cv_prompt_nvidia
    )
from .models import (
    get_open_ai_json,
    get_nvdia,
    get_ollama,
    get_open_ai
)


# Logging configuration
logger = logging.getLogger(__name__)


def get_chain( 
                prompt_template: str, 
                get_model: callable = get_nvdia,
                temperature : float = 0.0,
                parser: JsonOutputParser = JsonOutputParser
              ) -> LLMChain:
    """Retorna la langchain chain"""
    if not prompt_template and not isinstance(prompt_template,PromptTemplate):
      raise LangChainError()
    
    logger.info(f"Initializing LangChain using : {get_model.__name__}")
    model = get_model(temperature=temperature)
    chain = prompt_template | model | parser()
    
    return chain