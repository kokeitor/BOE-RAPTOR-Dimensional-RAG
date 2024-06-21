import logging
import logging.config
import logging.handlers
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from GRAPH_RAG.base_models import (
    State,
    Analisis
)
from GRAPH_RAG.config import ConfigGraph
from GRAPH_RAG.chains import get_chain
from GRAPH_RAG.nodes import (
    retriever,
    retreived_docs_grader,
    route_generate_requery,
    process_query,
    generator,
    hallucination_checker,
    generation_grader,
    final_report,
    route_generate_grade_gen,
    route_generate_final
)

logger = logging.getLogger(__name__)

def create_graph(config : ConfigGraph) -> StateGraph:
    
    graph = StateGraph(State)
    
    vector_db = config.vector_db
    docs_grader = config.agents.get("retreived_docs_grader",None)
    query_processor = config.agents.get("reprocess_query",None)
    generator_agent = config.agents.get("generator",None)
    hallucination_grader = config.agents.get("hallucination_checker",None)
    answer_grader = config.agents.get("generation_grader",None)


    # Define the nodes
    graph.add_node("retriever",lambda state: retriever(state=state,vector_database=vector_db)) 
    graph.add_node("retreived_docs_grader",lambda state: retreived_docs_grader(state=state,agent=docs_grader, get_chain=get_chain))
    graph.add_node("reprocess_query",lambda state: process_query(state=state,agent=query_processor, get_chain=get_chain))
    graph.add_node("generator",lambda state: generator(state=state,agent=generator_agent, get_chain=get_chain))
    graph.add_node("hallucination_checker",lambda state: hallucination_checker(state=state,agent=hallucination_grader, get_chain=get_chain))
    graph.add_node("generation_grader",lambda state: generation_grader(state=state,agent=answer_grader, get_chain=get_chain))
    graph.add_node("final_report",lambda state: final_report(state=state))

    # Add edges to the graph
    graph.set_entry_point("retriever")
    graph.set_finish_point("final_report")
    graph.add_edge("retriever", "retreived_docs_grader")
    graph.add_conditional_edges( "retreived_docs_grader",lambda state: route_generate_requery(state=state))
    graph.add_edge("reprocess_query", "retriever")
    graph.add_edge( "generator","hallucination_checker")
    graph.add_conditional_edges( "hallucination_checker",lambda state: route_generate_grade_gen(state=state))
    graph.add_conditional_edges( "generation_grader",lambda state: route_generate_final(state=state))
    graph.add_edge("final_report",END)
    
    return graph

def compile_workflow(graph):
    workflow = graph.compile()
    return workflow