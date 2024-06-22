import logging
import os
import sys
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from GRAPH_RAG.models import get_open_ai_json
from GRAPH_RAG.config import ConfigGraph
from GRAPH_RAG.chains import get_chain
from GRAPH_RAG.base_models import Question
from GRAPH_RAG.graph import create_graph, compile_workflow
from GRAPH_RAG.prompts import question_chat_history_prompt
from GRAPH_RAG.graph_utils import get_id, get_current_spanish_date_iso


# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Loading env variables
load_dotenv()


# Logger initializer
logger = logging.getLogger(__name__)


# app config
st.set_page_config(page_title="Streamlit Boe Chatbot", page_icon="🤖")
st.title("Chat BOEt")

def get_response(
             user_query : str, 
             chat_history : str, 
             model : callable = get_open_ai_json, 
             prompt : ChatPromptTemplate = question_chat_history_prompt,
             ):
    
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..','..','config/graph', 'graph.json') 
    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'config/graph', 'querys.json') 
        
    logger.info(f"{DATA_PATH=}")
    logger.info(f"{CONFIG_PATH=}")
    logger.info(f"Graph mode")
    logger.info(f"Getting Data and Graph configuration from {DATA_PATH=} and {CONFIG_PATH=} ")
    config_graph = ConfigGraph(config_path=CONFIG_PATH, data_path=DATA_PATH)
        
    logger.info("Creating graph and compiling workflow...")
    graph = create_graph(config=config_graph)
    compiled_graph = compile_workflow(graph)
    logger.info("Graph and workflow created")
    
    thread = {"configurable": {"thread_id": config_graph.thread_id}}
    iteraciones = {"recursion_limit": config_graph.iteraciones}
     
    chain = get_chain(prompt_template=prompt, get_model=model, temperature=0, parser=JsonOutputParser())
    response = chain.invoke({"chat_history":chat_history, "user_question":user_query})
    human_question = Question(id=get_id(), user_question= response["new_user_question"], date=get_current_spanish_date_iso())
    
    logger.info(f"User Question: {human_question.user_question}")
    logger.info(f"User id question: {human_question.id}")
    inputs = {"question": [f"{human_question.user_question}"], "date" : human_question.date}
    
    return compiled_graph.stream(inputs, iteraciones)

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

    
# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        ai_response = st.write_stream(get_response(user_query, st.session_state.chat_history))
        # st.write(ai_response)

    st.session_state.chat_history.append(AIMessage(content=ai_response))
