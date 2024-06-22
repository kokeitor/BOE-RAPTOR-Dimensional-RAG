from typing import Union, Optional, Callable, ClassVar, TypedDict, Annotated
from pydantic import BaseModel
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser, BaseTransformOutputParser
from langchain.prompts import PromptTemplate
from dataclasses import dataclass
from VectorDB.db import get_chromadb_retriever, get_pinecone_retriever, get_qdrant_retriever
from GRAPH_RAG.models import get_hg_emb
import operator
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.vectorstores import VectorStore


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
    date : str
    

class State(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: user question
        generation: LLM generation
        query_process: 'yes' 'no' -> reprocess or not the user query 
        documents: list of documents retrieved
        fact_based_answer : 'yes' 'no' -> LLM generation based on document retrieved (analog to hallucination : 'no' or 'yes')
        useful_answer : 'yes' 'no' -> LLM generation answer respond or not to the question 
        final_report 
        
    """
    question : Annotated[list[str],operator.add]
    generation : str
    query_process : str
    documents : Union[list[str],None] = None
    fact_based_answer : str
    useful_answer : int
    report : str
    
@dataclass()  
class Agent:
    agent_name : str
    model : str
    get_model : Callable
    temperature : float
    prompt : PromptTemplate
    parser : BaseTransformOutputParser
    
@dataclass()  
class VectorDB:
    client : str
    hg_embedding_model : str
    k : int
        
    def get_retriever_vstore(self) -> tuple[VectorStoreRetriever,VectorStore]:
        if self.client == 'pinecone':
            return  get_pinecone_retriever(
                                    embedding_model=self.hg_embedding_model, 
                                    search_kwargs={"k" : self.k}
            )
        elif self.client == 'chromadb':
            return get_chromadb_retriever(
                                    embedding_model=self.hg_embedding_model, 
                                    search_kwargs = {"k" : self.k}
            )
        elif self.client == 'qdrant':
            return get_qdrant_retriever(
                                    embedding_model=self.hg_embedding_model, 
                                    search_kwargs = {"k" : self.k}
            )

