import logging
from termcolor import colored
from typing import Dict, List, Tuple, Union, Optional, Callable
from langchain.prompts import PromptTemplate
from pydantic import ValidationError
from base_models import (
    State,
    Analisis,
    Candidato,
    Agent
)
from prompts import (
    analyze_cv_prompt,
    offer_check_prompt,
    re_analyze_cv_prompt,
    cv_check_prompt,
    analyze_cv_prompt_nvidia
    )
from models import (
    get_open_ai_json,
    get_nvdia,
    get_ollama,
    get_open_ai
)
from .chains import get_chain
from .graph_utils import get_current_spanish_date_iso, merge_page_content


# Logging configuration
logger = logging.getLogger(__name__)


"""
### Router chain
router_chain = routing_prompt | llm | JsonOutputParser()

### Grader chain
grader_chain = grader_prompt | llm | JsonOutputParser()

### RAG chain (generation)
rag_chain = gen_prompt | gen_llm | StrOutputParser()

### Hallucination chain (grader)
hallucination_chain = hallucination_prompt | llm | JsonOutputParser()

### Answer grader
answer_chain = answer_prompt | llm | JsonOutputParser()

### calsifier grader
clasify_chain = clasify_prompt | llm | JsonOutputParser() 
"""

### Nodes
def retriever(retriever, state : State) -> State:
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    logger.info(f"Retriever node : \n {state}")
    question = state["question"]
    documents = retriever.invoke(question)
    logger.info(f"Number of retrieved docs : {len(documents)}")
    logger.debug(f"Retrieved documents : \n {documents}")
    state["documents"] = documents

    return state


def retreived_docs_grader(state : State, agent : Agent, get_chain : Callable = get_chain) -> State:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run [?] IMPLEMENATCION LOGICA DE ESTE NODO

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    logger.info(f"Retrieved Documents Grader Node : \n {state}")
    question = state["question"]
    documents = state["documents"]
    
    # Grader chain
    grader_chain = get_chain(get_model=agent.get_model, prompt_template=agent.prompt, temperature=agent.temperature)
    
    # Score each doc
    relevant_docs = []
    for index_doc , d in enumerate(documents):
        content = d.page_content
        logger.info(f"Document content : \n {content}")
        
        score = grader_chain.invoke({"question": question, "document": content})
        grade = score['score']
        
        # Document relevant
        if grade.lower() == "yes":
            logger.info(f"--- GRADE: DOCUMENT {index_doc} -> RELEVANT---")
            relevant_docs.append(d)
        # Document not relevant
        else:
            logger.warning(f"--- GRADE: DOCUMENT {index_doc} -> NOT RELEVANT")
            
    # if only 0 or 1 doc relevant -> query processing necesary [no enough retrieved relevant context to answer]
    if len(relevant_docs) == 0:  
        state["query_reprocess"] = 'yes'
        state["documents"] = None
    else:
        state["query_reprocess"] = 'no'
        state["documents"] = relevant_docs

    return state

def generator(state : State, agent : Agent, get_chain : Callable = get_chain) -> State:
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    logger.info(f"RAG Generator node : \n {state}")
    question = state["question"]
    
    # Get the merge context from retrieved docs
    documents = state["documents"]
    context = merge_page_content(docs = documents) # Merge docs page_content into unique str for the model context
    
    # RAG generation
    rag_chain = get_chain(get_model=agent.get_model, prompt_template=agent.prompt, temperature=agent.temperature)
    generation = rag_chain.invoke({"context": context, "question": question})
    logger.info(f"RAG Context : \n {context}")
    logger.info(f"RAG Question : \n {question}")
    logger.info(f"RAG Response : \n {generation}")
    
    # Update Graph State
    state["generation"] = generation
    
    return state


def _web_search(state : State, agent : Agent, get_chain : Callable = get_chain) -> State:
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return state


def reprocess_query(state : State, agent : Agent, get_chain : Callable = get_chain) -> State:
    """
    Query processing tool

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    logger.info(f"Query Reprocessing : \n {state}")
    
    question = state["question"]
    documents = state["documents"]
    ### here code for procesing query 
    ### ...
    ###
    new_question= "¿A quien se promueve como magistrada en el Puerto de la Cruz?"
    state["question"] = new_question
    logger.info(f"{question=} //after reprocessing// {new_question=}")

    return state

def final_report(state:State) -> State:

    analisis_final = state["analisis"][-1]
    candidato = state["candidato"]
    
    logger.info(f"Analisis final : \n {analisis_final}")
    print(colored(f"\nReporte final 📝\n\nFecha del analisis : {analisis_final.fecha}\n\n**CANDIDATO**\n{candidato.cv}\n\n**OFERTA**\n{candidato.oferta}\n\n**ANALISIS**\n- Puntuacion : {analisis_final.puntuacion}\n- Experiencias : {analisis_final.experiencias}\n- Descripcion : {analisis_final.descripcion}", 'light_yellow',attrs=["bold"]))

    state = {**state, "analisis_final": analisis_final}

    return state

def end_node(state:State) -> State:
    logger.info(f"Nodo final")
    return state


### Conditional edge functions
def route_question(state : State, agent : Agent, get_chain : Callable = get_chain) -> str:
    """
    Route question to question processing tool or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    logger.info(f"Router Query : \n {state}")
    question = state["question"]
    
    router_chain = get_chain(get_model=agent.get_model, prompt_template=agent.prompt, temperature=agent.temperature)
    source = router_chain.invoke({"question": question})  
    next_node = source["source"]
    logger.info(f"Routing query to -> \n {next_node=}")
    
    # Check model output format error
    if next_node == 'query_reprocess':
        return "query_reprocess"
    elif next_node == 'vectorstore':
        return "vectorstore"
    else:
        logger.exception(f"Unkonwn response : {next_node=} from the route_question model node")
        raise ValueError(f"Unkonwn response : {next_node=} from the route_question model node")

def decide_to_generate(state : State) -> str:
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    logger.info(f"Decide to generate edge or query tool: \n {state}")
    query_reprocess = state["query_reprocess"]
    if query_reprocess == "yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        logger.info("Decision ->  Query Reprocess")
        return "query_reprocess"
    if query_reprocess == "no":
        # We have relevant documents, so generate answer
        logger.info("Decision -> Generation")
        return "generate"


def check_generation(state : State, agent : Agent, get_chain : Callable = get_chain) -> str:
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    logger.info(f"Generation hallucionation checker : \n {state=}")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    hallu_chain = get_chain(get_model=agent.get_model, prompt_template=agent.prompt, temperature=agent.temperature)
    response = hallu_chain.invoke({"documents": documents, "generation": generation})
    grade = response['score']
    logger.info(f"Hallucionation grade-> {grade=}")

    # Check hallucination
    if grade == "no":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        logger.debug(f"Generation is grounded in documents")
        return 'generation_grader'
    elif grade == "yes":
        return "retriever"
    else:
        logger.exception(f"Unkonwn response : {grade=} from the check generation node")
        raise ValueError(f"Unkonwn response : {grade=} from the check generation node")
    
def grade_generation(state : State, agent : Agent, get_chain : Callable = get_chain, threshold : float = 0.7) -> str:
    """
    Grades the accuracy/precison of the model response whrn answers a given question

    Args:
        state (State): _description_
        agent (Agent): _description_
        get_chain (Callable, optional): _description_. Defaults to get_chain.

    Returns:
        str: _description_
    """
    logger.info(f"Grade Generation node : \n {state=}")
    question = state["question"]
    generation = state["generation"]
    
    garder_chain = get_chain(get_model=agent.get_model, prompt_template=agent.prompt, temperature=agent.temperature)
    response = garder_chain.invoke({"question": question, "generation": generation})
    logger.info(f"Grade of the generation-> {response=}")
    
    try:
        grade = response['score']
    except:
        logger.exception(f"Error in the grade generation json format, not 'score' key")
    
    if not isinstance(grade, (int,float)):
        logger.exception(f"Error in the grade generation format, not a valid type {type(grade)}")
        raise ValueError(f"Error in the grade generation format, not a valid type {type(grade)}")
    
    # threshold logic
    if grade < threshold:
        logger.exception(f"No good enough generation, back to genereation node")
        logger.debug(f"Generation is grounded in documents")
        return 'generation'
    elif grade >= threshold:
        logger.exception(f"Good response -> final_report")
        return "final_report"
    else:
        logger.exception(f"Unkonwn response : {grade=} from the grader generation node")
        raise ValueError(f"Unkonwn response : {grade=} from the grader generation node")

