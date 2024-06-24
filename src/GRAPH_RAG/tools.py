from langchain.tools import BaseTool, StructuredTool, tool
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.pydantic_v1 import BaseModel, Field
from langchain.agents import AgentExecutor, create_react_agent
    
    
@tool("search")
def search_tool(query: str):
    """Searches for information on The Spanish BOE (Boletín Oficial del Estado) 
    is the official state gazette of Spain. It publishes legal documents, including laws,
    decrees, official announcements, and government resolutions. The BOE ensures the 
    dissemination and transparency of Spanish legislative and administrative actions, making them accessible to the public.
    Cannot be used to research any other topics. Search query must be provided
    in natural language and be verbose."""
    # this is a "RAG" emulator
    return "Boe extra info ... : "


@tool("final_answer_tool")
def final_answer_tool(
    answer: str
):
    """Returns a natural language response to the user in `answer`
    """
    return ""