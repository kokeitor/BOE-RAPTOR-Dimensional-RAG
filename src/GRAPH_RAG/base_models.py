from typing import Union, Optional, Callable, ClassVar, TypedDict, Annotated
from pydantic import BaseModel
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser, BaseTransformOutputParser
from langchain.prompts import PromptTemplate
from dataclasses import dataclass
import operator


class Analisis(BaseModel):
    id : str
    fecha : str
    puntuacion: int
    experiencias: list[dict[str,str]]
    descripcion: str
    status: str
    
class Question(BaseModel):
    id : Union[str,None] = None
    user_question : str

class State(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents 
        hallucination
        answer_grade 
        final_report 
        
    """
    question : Annotated[str,operator.add]
    generation : str
    query_process : str
    documents : Union[list[str],None] = None
    fact_based_answer : str
    useful_answer : int
    final_report : str
    
@dataclass()  
class Agent:
    agent_name : str
    model : str
    get_model : Callable
    temperature : float
    prompt : PromptTemplate
    parser : BaseTransformOutputParser
