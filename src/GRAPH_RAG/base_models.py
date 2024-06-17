from typing import Union, Optional, Callable, ClassVar, TypedDict, Annotated
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from dataclasses import dataclass


class Analisis(BaseModel):
    id : str
    fecha : str
    puntuacion: int
    experiencias: list[dict[str,str]]
    descripcion: str
    status: str
    
class Candidato(BaseModel):
    id : Union[str,None] = None
    cv : str
    oferta : str

class State(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents 
    """
    question : str
    generation : str
    query_process : str
    documents : Union[list[str],None] = None
    
@dataclass()  
class Agent:
    agent_name : str
    model : str
    get_model : Callable
    temperature : float
    prompt : PromptTemplate

### State
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents 
    """
    question : str
    generation : str
    query_processing : str
    documents : list[str]