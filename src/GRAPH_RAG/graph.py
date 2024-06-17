import logging
import logging.config
import logging.handlers
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage
from base_models import (
    State,
    Candidato,
    Analisis
)
from nodes import (
    retriever,
    generator,
    query_tool,
    final_report,
    end_node,
)

logger = logging.getLogger(__name__)

def create_graph(config : ConfigGraph) -> StateGraph:
    
    graph = StateGraph(GraphState)

    # Define the nodes
    graph.add_node("query_tool", query_tool) # web search
    graph.add_node("retrieve", retriever) # retrieve
    graph.add_node("grade_documents", grade_documents) # grade documents
    graph.add_node("generate", generator) # generatae
    
    analyzer = config.agents.get("analyzer",None)
    re_analyzer = config.agents.get("re_analyzer",None)
    cv_reviewer = config.agents.get("cv_reviewer",None)
    offer_reviewer = config.agents.get("offer_reviewer",None)

    graph = StateGraph(State)
    graph.add_node("analyzer",lambda state: analyzer_agent(state=state,analyzer=analyzer, re_analyzer=re_analyzer))
    graph.add_node("reviewer_cv",lambda state: reviewer_cv_agent(state=state, agent=cv_reviewer))
    graph.add_node( "reviewer_offer", lambda state: reviewer_offer_agent(state=state, agent=offer_reviewer))
    graph.add_node( "report", lambda state: final_report(state=state))
    graph.add_node( "end_node", lambda state: end_node(state=state))

    # Add edges to the graph
    graph.set_entry_point("analyzer")
    graph.set_finish_point("end_node")
    graph.add_edge("analyzer", "reviewer_cv")
    graph.add_conditional_edges( "reviewer_cv",lambda state: pass_cv_review(state=state),)
    graph.add_conditional_edges( "reviewer_offer",lambda state: pass_offer_review(state=state))
    graph.add_edge("report","end_node")

    return graph

def compile_workflow(graph):
    workflow = graph.compile()
    return workflow