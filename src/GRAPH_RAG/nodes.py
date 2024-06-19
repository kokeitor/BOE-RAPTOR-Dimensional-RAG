import logging
from termcolor import colored
from typing import Dict, List, Tuple, Union, Optional, Callable
from langchain.prompts import PromptTemplate
from pydantic import ValidationError
from GRAPH_RAG.base_models import (
    State,
    Analisis,
    Agent
)
from GRAPH_RAG.models import (
    get_open_ai_json,
    get_nvdia,
    get_ollama
)
from GRAPH_RAG.chains import get_chain
from GRAPH_RAG.graph_utils import get_current_spanish_date_iso, merge_page_content


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
def retriever(retrievers : list , state : State) -> State:
    """Retrieve documents from vector database"""
    
    logger.info(f"Retriever node : \n {state}")
    question = state["question"]
    documents = [retriever.invoke(question) for retriever in retrievers]
    logger.info(f"Number of retrieved docs : {len(documents)}")
    logger.debug(f"Retrieved documents : \n {documents}")
    state["documents"] = documents

    return state


def retreived_docs_grader(state : State, agent : Agent, get_chain : Callable = get_chain) -> State:
    """Determines whether the retrieved documents are relevant to the question"""
    
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
    """Generate answer using RAG on retrieved documents"""
    
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


def process_query(state : State, agent : Agent, get_chain : Callable = get_chain) -> State:
    """Reprocess a user query to improve docs retrieval"""

    logger.info(f"Query Reprocessing : \n {state}")
    question = state["question"]
    chain = get_chain(get_model=agent.get_model, prompt_template=agent.prompt, temperature=agent.temperature)
    response = chain.invoke({"question": question})
    reprocesed_question = response["reprocess_question"]
    state["question"] = reprocesed_question
    logger.info(f"{question=} // after reprocessing question -> {response=}")

    return state


def hallucination_checker(state : State, agent : Agent, get_chain : Callable = get_chain) -> State:
    """Checks for hallucionation on the response or generation"""

    logger.info(f"hallucination_checker node : \n {state}")
    generation = state["question"]
    documents = state["documents"]
    context = merge_page_content(docs = documents) # Merge docs page_content into unique str for the model context
    
    hall_chain = get_chain(get_model=agent.get_model, prompt_template=agent.prompt, temperature=agent.temperature)
    response = hall_chain.invoke({"documents": context, "generation": generation})
    fact_based_answer = response["score"]
    logger.info(f"hallucination grade : {response=}")
    
    # Update Graph State
    state["fact_based_answer"] = fact_based_answer

    return state

def generation_grader(state : State, agent : Agent, get_chain : Callable = get_chain) -> State:
    """Grades the generation/answer given a question"""
    
    logger.info(f"generation_grader node : \n {state}")
    generation = state["question"]
    question = state["question"]

    garder_chain = get_chain(get_model=agent.get_model, prompt_template=agent.prompt, temperature=agent.temperature)
    response = garder_chain.invoke({"question": question, "generation": generation})
    grade = response["score"]
    logger.info(f"Answer grade : {response=}")
    
    # Update Graph State
    state["useful_answer"] = grade

    return state

def final_report(state:State) -> State:

    generation = state["generation"]
    question = state["question"]
    documents = state["documents"]
    grade = state["answer_grade"]
    
    logger.info(f"Final model response : \n {state}")
    print(colored(f"\nFinal model report 📝\n\**QUESTION**: {question}\n\n**RETRIEVED DOCS**\n{documents}\n\n**ANSWER**\n{generation}\n\n**ANSWER GRADE** : {grade}", 'light_yellow',attrs=["bold"]))

    return state


### Conditional edge functions
def route_generate_requery(state : State) -> str:
    """Route to generation or to reprocess question """
    
    logger.info(f"Router Generation or Reprocess Query : \n {state}")
    query_reprocess = state["query_reprocess"]
    filter_docs = state["documents"]
    
    if query_reprocess == "yes" and filter_docs is None:
        logger.info(f"Routing to -> 'query_reprocess'")
        return 'query_reprocess'
    if query_reprocess == "no" and filter_docs is not None:
        logger.info(f"Routing to -> 'generate'")
        return 'generate'
    
def route_generate_grade_gen(state : State) -> str:
    """Route to generation or to grade the generation/answer"""
    
    logger.info(f"Router Generation or Grader Generation : \n {state}")
    fact_based_answer = state["fact_based_answer"]
    
    if fact_based_answer == "yes":
        logger.info(f"Routing to -> 'Grader generation'")
        return 'generation_grader'
    if fact_based_answer == "no":
        logger.info(f"Routing to -> 'Generation'")
        return 'generator'

    
def route_generate_final(state : State) -> str:
    """Route to generation or to final report"""
    
    logger.info(f"Router Generation or Final report : \n {state}")
    useful_answer = state["useful_answer"]
    
    if useful_answer == "yes":
        logger.info(f"Routing to -> 'Final Report'")
        return 'final_report'
    if useful_answer == "no":
        logger.info(f"Routing to -> 'Generation'")
        return 'generator'

    
def _grade_generation(state : State, agent : Agent, get_chain : Callable = get_chain, threshold : float = 0.7) -> str:
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


