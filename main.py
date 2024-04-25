import os 
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import List
from chains import router_chain,grader_chain, rag_chain, hallucination_chain,answer_chain
from package.utils import translate,format_docs
from langchain.schema import Document
from tqdm import tqdm
from langgraph.graph import END, StateGraph
from package.db import db_conexion

### env keys
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
load_dotenv()



## RETRIEVER FUNCTION 
chroma_vectorstore,retriever_chroma,pinecone_vectorstore,retriever_pinecone = db_conexion()
def docs_from_retriver(question :str):
    
    try: 
        return retriever_chroma.invoke(question)
    except Exception as e:
        print(f"{e}")

    try: 
        return retriever_pinecone.invoke(question)
    except Exception as e:
        print(f"{e}")





### State
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents 
    """
    question : str
    generation : str
    query_processing : str
    documents : List[str]


### Nodes
def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    
    documents = docs_from_retriver(question=question)
    return {"documents": documents, "question": question}

def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # Translation of query and docs
    #question_trl = translate(text = question, target_lang = "EN-GB" , verbose  = 0, mode = "LOCAL_LLM")
    #docs_trl  = [Document(page_content = translate(text = doc_trl, target_lang = "EN-GB" , verbose  = 0, mode = "LOCAL_LLM")) for doc_trl in documents]
    
    # Format docs obj into unique str for model
    format_doc_text = format_docs(docs = documents )
    
    # RAG generation
    generation = rag_chain.invoke({"context": format_doc_text, "question": question})
    print("generation", generation)
    print("context", format_doc_text)
    print("question", question)
    #gen_trl = translate(text = "generation", generation, target_lang = "ES" , verbose  = 0, mode = "LOCAL_LLM")
    
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    
    # Translation of query and docs
    #question_trl = translate(text = question, target_lang = "EN-GB" , verbose  = 0, mode = "LOCAL_LLM")
    #docs_trl  = [Document(page_content = translate(text = doc_trl, target_lang = "EN-GB" , verbose  = 0, mode = "LOCAL_LLM")) for doc_trl in documents]
    
    # Score each doc
    filtered_docs = []
    query_tool = "No"
    for index_doc, d in enumerate(documents):
        score = grader_chain.invoke({"question": question, "document": d.page_content})
        grade = score['score']
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print(f"---GRADE: DOCUMENT {index_doc} NOT RELEVANT---")
            # We do not include the document in filtered_docs

    # if only 0 or 1 doc relevant -> query processing necesary [no enough retrieved relevant context to answer]
    if len(filtered_docs) <= 1:  
        query_tool = "Yes"

    return {"documents": filtered_docs, "question": question, "query_processing": query_tool}


#def web_search(state):
#    """
#    Web search based based on the question
#
#    Args:
#        state (dict): The current graph state
#
#    Returns:
#        state (dict): Appended web results to documents
#    """
#
#    print("---WEB SEARCH---")
#    question = state["question"]
#    documents = state["documents"]
#
#    # Web search
#    docs = web_search_tool.invoke({"query": question})
#    web_results = "\n".join([d["content"] for d in docs])
#    web_results = Document(page_content=web_results)
#    if documents is not None:
#        documents.append(web_results)
#    else:
#        documents = [web_results]
#    return {"documents": documents, "question": question}


def query_tool(state) -> dict:
    """
    Query processing tool

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---QUERY PROCESSING---")
    question = state["question"]
    documents = state["documents"]
    ### here code for procesing query ...
    ###
    for _ in tqdm(range(4)):
        print("PROCESSING THE QUERY ... ")
        
    question = "¿Cual es la duración total de las enseñanzas en ciclos de grado medio?"
        
    return {"documents": documents, "question": question}

### Conditional edge

def route_question(state) -> str:
    """
    Route question to question processing tool or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    #question_trl = translate(text = question, target_lang = "EN-GB" , verbose  = 0, mode = "LOCAL_LLM")
    
    source = router_chain.invoke({"question": question})  
    print(source)
    print(source['datasource'])
    if source['datasource'] == 'query_tool':
        print("---ROUTE QUESTION TO QUERY PROCESSING TOOL---")
        return "query_tool"
    elif source['datasource'] == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    query_tool = state["query_processing"]
    filtered_documents = state["documents"]

    if query_tool == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ONLY 1 OR 0 DOCUMENTS ARE RELEVANT TO QUESTION, QUERY PROCESSING NECESARY---")
        return "query_tool"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

### Conditional edge

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    # Translation of query and docs
    #question_trl = translate(text = question, target_lang = "EN-GB" , verbose  = 0, mode = "LOCAL_LLM")
    #gen_trl = translate(text = question, target_lang = "EN-GB" , verbose  = 0, mode = "LOCAL_LLM")
    #docs_trl  = [Document(page_content = translate(text = doc_trl, target_lang = "EN-GB" , verbose  = 0, mode = "LOCAL_LLM")) for doc_trl in documents]

    score = hallucination_chain.invoke({"documents": documents, "generation": generation})
    grade = score['score']

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_chain.invoke({"question": question,"generation": generation})
        grade = score['score']
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("query_tool", query_tool) # web search
workflow.add_node("retrieve", retrieve) # retrieve
workflow.add_node("grade_documents", grade_documents) # grade documents
workflow.add_node("generate", generate) # generatae

# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "query_tool": "query_tool",
        "vectorstore": "retrieve",
    },
)

workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "query_tool": "query_tool",
        "generate": "generate",
    },
)
workflow.add_edge("query_tool", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": END,
    },
)

# Compile
app = workflow.compile()

# Test
if __name__ == '__main__':
    
    user_question = input("¡PREGUNTA AL BOE!")
    inputs = {"question": f"{user_question}"}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Finished running: {key}:")
    print("BOE DICE : " , value["generation"])