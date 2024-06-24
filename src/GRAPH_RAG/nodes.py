import logging
from termcolor import colored
from typing import Dict, List, Tuple, Union, Optional, Callable
from langchain.prompts import PromptTemplate
from pydantic import ValidationError
from GRAPH_RAG.base_models import (
    State,
    VectorDB,
    Agent
)
from GRAPH_RAG.chains import get_chain
from GRAPH_RAG.agents import get_openai_agent
from GRAPH_RAG.graph_utils import get_current_spanish_date_iso, merge_page_content


# Logging configuration
logger = logging.getLogger(__name__)


### Nodes

def process_user_input_agent(state : State, get_agent : callable = get_openai_agent ) -> dict:
    """Agent node that process a input user query and decides using a search boe info tool or to 
    generate a direct-raw response if the input is not boe related"""
    
    logger.info(f"Initial Agent node : \n {state}")
    print(colored(f"\nInitial Agent node : ",'light_green'))
    
    # Input user question
    user_query = state["user_input"]
    agent_intermediate_steps = state["agent_intermediate_steps"]
    
    # Input format for agent 
    inputs = {
    "user_query": str(user_query),
    "agent_intermediate_steps": [agent_intermediate_steps]
        }

    agent = get_agent()
    agent_response = agent.invoke(inputs)
    
    logger.info(f"Agent Response : \n {agent_response}")
    
    print(colored(f"\user_query -> {state['user_query'][-1]}\nAgent Response -> {agent_response}\n",'light_green'))

    return {"agent_output" : agent_response}


def retriever(vector_database : VectorDB, state : State) -> dict:
    """Retrieve documents from vector database"""
    
    logger.info(f"Retriever node : \n {state}")
    print(colored(f"\nRetriever node 👩🏿‍💻 ",'light_blue',attrs=["bold"]))
    
    retriever_vdb , _ = vector_database.get_retriever_vstore()
    logger.info(f"Using client for retrieval : {vector_database.client=}")
    
    question = state["question"][-1]
    documents = retriever_vdb.invoke(question)
    
    logger.info(f"Number of retrieved docs : {len(documents)}")
    logger.debug(f"Retrieved documents : \n {documents}")
    
    print(colored(f"Date = {state['date']}\nQuestion = {state['question']}\nNumber of retrieved docs =  {len(documents)}",'light_blue',attrs=["bold"]))

    return {"documents": documents}


def retreived_docs_grader(state : State, agent : Agent, get_chain : Callable = get_chain) -> dict:
    """Determines whether the retrieved documents are relevant to the question"""
    
    logger.info(f"Retrieved Documents Grader Node : \n {state}")
    print(colored(f"\n{agent.agent_name=} 👩🏿 -> {agent.model=}",'magenta',attrs=["bold"]))
    
    question = state["question"][-1]
    documents = state["documents"]
    
    # Grader chain
    grader_chain = get_chain(get_model=agent.get_model, prompt_template=agent.prompt, temperature=agent.temperature, parser=agent.parser)
    
    # Score each doc
    relevant_docs = []
    for index_doc , d in enumerate(documents):
        content = d.page_content
        logger.info(f"Document content : \n {content}")

        score = grader_chain.invoke({"question": question, "document": content})
        grade = score['score']
        
        print(colored(f"\nDoc {index_doc} -- {score=}\nScored Doc content : {content}",'magenta',attrs=["bold"]))
        
        # Document relevant
        if grade.lower() == "yes":
            logger.info(f"GRADE: DOCUMENT {index_doc} as RELEVANT")
            relevant_docs.append(d)
        # Document not relevant
        else:
            logger.warning(f"GRADE: DOCUMENT {index_doc} as NOT RELEVANT")
               
    # if only 0 or 1 doc relevant -> query processing necesary [no enough retrieved relevant context to answer]
    if len(relevant_docs) == 0:  
        return {"documents": None, "query_reprocess" : 'no'}
        
    else:
        return {"documents": relevant_docs, "query_reprocess" : 'no'}
    

def generator(state : State, agent : Agent, get_chain : Callable = get_chain) -> dict:
    """Generate answer using RAG on retrieved documents"""
    
    logger.info(f"RAG Generator node : \n {state}")
    print(colored(f"\n{agent.agent_name=} 👩🏽 -> {agent.model=} : ", 'light_red',attrs=["bold"]))
        
    question = state["question"][-1]
    
    # Get the merge context from retrieved docs
    documents = state["documents"]
    context = merge_page_content(docs = documents) # Merge docs page_content into unique str for the model context
    
    # RAG generation
    rag_chain = get_chain(get_model=agent.get_model, prompt_template=agent.prompt, temperature=agent.temperature, parser=agent.parser)
    generation = rag_chain.invoke({"context": context, "question": question})
    answer = generation["answer"]
    logger.info(f"RAG Context : \n {context}")
    logger.info(f"RAG Question : \n {question}")
    logger.info(f"RAG Response : \n {generation}")
    
    print(colored(f"\nQuestion -> {state['question'][-1]}\nResponse -> {generation}\n",'light_red',attrs=["bold"]))
    
    return {"generation" : answer}


def process_query(state : State, agent : Agent, get_chain : Callable = get_chain) -> dict:
    """Reprocess a user query to improve docs retrieval"""

    logger.info(f"Query Reprocessing : \n {state}")
    print(colored(f"\n{agent.agent_name=} 📝 -> {agent.model=} : ", 'light_yelow',attrs=["bold"]))
    
    question = state["question"][-1]
    chain = get_chain(get_model=agent.get_model, prompt_template=agent.prompt, temperature=agent.temperature ,parser=agent.parser)
    response = chain.invoke({"question": question})
    reprocesed_question = response["reprocess_question"]

    logger.info(f"{question=} // after reprocessing question -> {response=}")
    print(colored(f"Initial question : {question=}\nAfter reprocessing question : {response=}",'light_yelow',attrs=["bold"]))
    
    return {"question" : reprocesed_question}


def hallucination_checker(state : State, agent : Agent, get_chain : Callable = get_chain) -> dict:
    """Checks for hallucionation on the response or generation"""

    logger.info(f"hallucination_checker node : \n {state}")
    print(colored(f"\n{agent.agent_name=} 👩🏿 -> {agent.model=} : ", 'light_cyan',attrs=["bold"]))
    
    generation = state["question"][-1]
    documents = state["documents"]
    context = merge_page_content(docs = documents) # Merge docs page_content into unique str for the model context
    
    hall_chain = get_chain(get_model=agent.get_model, prompt_template=agent.prompt, temperature=agent.temperature, parser=agent.parser)
    response = hall_chain.invoke({"documents": context, "generation": generation})
    fact_based_answer = response["score"]
    logger.info(f"hallucination grade : {response=}")

    print(colored(f"Answer supported by context -> {response}",'light_cyan',attrs=["bold"]))
    
    return {"fact_based_answer" : fact_based_answer}


def generation_grader(state : State, agent : Agent, get_chain : Callable = get_chain) -> dict:
    """Grades the generation/answer given a question"""
    
    logger.info(f"generation_grader node : \n {state}")
    print(colored(f"\n{agent.agent_name=} 👩🏿 -> {agent.model=} : ", 'light_cyan',attrs=["bold"]))
     
    generation = state["generation"]
    question = state["question"][-1]

    garder_chain = get_chain(get_model=agent.get_model, prompt_template=agent.prompt, temperature=agent.temperature, parser=agent.parser)
    response = garder_chain.invoke({"question": question, "generation": generation})
    grade = response["score"]
    logger.info(f"Answer grade : {response=}")
    
    print(colored(f"Useful answer to resolve the question -> {response}",'light_cyan',attrs=["bold"]))
    
    return { "useful_answer" : grade}


def final_report(state : State) -> dict:
    """Returns and writes the final response of the model in a custom format"""

    generation = state["generation"]
    questions = state["question"]
    documents = state["documents"]
    grade_answer= state["useful_answer"]
    grade_hall= state["fact_based_answer"]
    query_process = state["query_process"]
    
    logger.info(f"Final model response : \n {state}")
    print(colored(f"\nFinal model report 📝\n\n**QUESTIONS**: {questions}\n\n**QUERY REPROCESS**{query_process}\n\n**RETRIEVED DOCS**\n{documents}\n\n**ANSWER**\n{generation}\n\n**CONTEXT BASED ANSWER GRADE** : {grade_hall}\n\n**ANSWER GRADE** : {grade_answer}", 'light_yellow',attrs=["bold"]))
   
    return {"report" : generation}


### Conditional edge functions
def route_retrieve_final(state : State) -> str:
    """Route to retrieve node or to final report"""
    
    logger.info(f"Router Retrieval or Final Report : \n {state}")

    if state["query_reprocess"] == "yes":
        logger.info("Routing to -> 'query_reprocess'")
        print(colored("\n\nRouting to -> query_reprocess\n\n",'light_green',attrs=["underline"]))
        return 'reprocess_query'
    if state["query_reprocess"] == "no":
        logger.info("Routing to -> 'generator'")
        print(colored("\n\nRouting to -> generator\n\n",'light_green',attrs=["underline"]))
        return 'generator'
    
def route_generate_requery(state : State) -> str:
    """Route to generation or to reprocess question """
    
    logger.info(f"Router Generation or Reprocess Query : \n {state}")

    if state["query_reprocess"] == "yes":
        logger.info("Routing to -> 'query_reprocess'")
        print(colored("\n\nRouting to -> query_reprocess\n\n",'light_green',attrs=["underline"]))
        return 'reprocess_query'
    if state["query_reprocess"] == "no":
        logger.info("Routing to -> 'generator'")
        print(colored("\n\nRouting to -> generator\n\n",'light_green',attrs=["underline"]))
        return 'generator'
    
def route_generate_grade_gen(state : State) -> str:
    """Route to generation or to grade the generation/answer"""
    
    logger.info(f"Router Generation or Grader Generation : \n {state}")

    if state["fact_based_answer"] == "yes":
        logger.info("Routing to -> 'Grader generation'")
        print(colored("\n\nRouting to -> Grader generation\n\n",'light_green',attrs=["underline"]))
        return 'generation_grader'
    if state["fact_based_answer"] == "no":
        logger.info("Routing to -> 'Generation'")
        print(colored("\n\nRouting to -> Generation\n\n",'light_green',attrs=["underline"]))
        return 'generator'

    
def route_generate_final(state : State) -> str:
    """Route to generation or to final report"""
    
    logger.info(f"Router Generation or Final report : \n {state}")
    
    if state["useful_answer"] == "yes":
        logger.info("Routing to -> 'Final Report'")
        print(colored("\n\nRouting to -> Final Report\n\n",'light_green',attrs=["underline"]))
        return 'final_report'
    if state["useful_answer"] == "no":
        logger.info("Routing to -> 'Generation'")
        print(colored("\n\nRouting to -> Generation\n\n",'light_green',attrs=["underline"]))
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
    question = state["question"][-1]
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


