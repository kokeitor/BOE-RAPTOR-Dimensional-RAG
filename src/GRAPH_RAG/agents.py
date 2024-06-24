from langchain.agents import create_openai_tools_agent
from langchain_core.prompts.chat import ChatPromptTemplate
from GRAPH_RAG.models import get_open_ai
from GRAPH_RAG.tools import search_tool,final_answer_tool
from GRAPH_RAG.prompts import agent_promt
from langchain_core.runnables.base import Runnable
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser


def get_openai_agent(
                    tools : list[callable] = [search_tool,final_answer_tool ], 
                    model : callable = get_open_ai, 
                    prompt : ChatPromptTemplate = agent_promt 
                    ) -> Runnable:
    
    agent = create_openai_tools_agent(
                                        llm=model(),
                                        tools=[tool for tool in tools],
                                        prompt=prompt
                                    )
    return agent 

def get_custom_openai_agent( 
                    tools : list[callable] = [search_tool,final_answer_tool], 
                    model : callable = get_open_ai, 
                    prompt : ChatPromptTemplate = agent_custom_prompt 
                    ) -> Runnable:
    
    llm_model = model()
    llm_with_tools = llm_model.bind_tools(tools)

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    return agent