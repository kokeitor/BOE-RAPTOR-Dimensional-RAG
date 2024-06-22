import logging
import os
import sys
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph.graph import CompiledGraph
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


def run_app(config_graph : ConfigGraph) -> None: 
    
    st.set_page_config(page_title="Streamlit Boe Chatbot", page_icon="🤖")
    st.title("Chat BOEt")

    def get_response(
                user_query : str, 
                chat_history : str, 
                model : callable = get_open_ai_json, 
                prompt : PromptTemplate = question_chat_history_prompt,
                config_graph : ConfigGraph = config_graph
                ):

        chain = get_chain(prompt_template=prompt, get_model=model, temperature=0, parser=JsonOutputParser)
        response = chain.invoke({"chat_history":chat_history, "user_question":user_query})
        human_question = Question(id=get_id(), user_question= response["new_user_question"], date=get_current_spanish_date_iso())
        
        logger.info(f"User Question: {human_question.user_question}")
        logger.info(f"User id question: {human_question.id}")
        inputs = {"question": [f"{human_question.user_question}"], "date" : human_question.date}
        
        return config_graph.compile_graph.stream(inputs, config_graph.iteraciones)


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
