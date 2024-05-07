from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from langchain_community.embeddings import GPT4AllEmbeddings 
from langchain.embeddings import HuggingFaceEmbeddings 


### LLM MODElS
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDING_MODEL_GPT4 = GPT4AllEmbeddings(model_name ="all‑MiniLM‑L6‑v2.gguf2.f16.gguf")
LOCAL_LLM = 'llama3'
llm = ChatOllama(model=LOCAL_LLM, format="json", temperature=0)
gen_llm = ChatOllama(model=LOCAL_LLM, temperature=0)


### PROMPTS 
routing_prompt_web_search = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
    user question to a vectorstore or web search. Use the vectorstore for questions on spanish BOE documents. \n
    You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
    or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
    no premable or explaination. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)

routing_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
    user question to a vectorstore or to a question processing tool. \n
    Use the vectorstore for questions related on spanish BOE documents. Use this list of topics that you can typically find in the BOE:
    Legislation, Regulations and Decrees, Government Announcements, Legal Notices, Public Employment, Judicial Appointments and Decisions, Economic and Financial Information,
    Grants and Subsidies, Sanctions, International Treaties, Intellectual Property Registrations, Company and Business Regulations and Awards and Honors. \n
    You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use the question processing tool. Give a binary choice 'query_tool' 
    or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
    no premable or explaination. 
    Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)

grader_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"])

gen_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)


hallucination_prompt = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
    an answer is grounded in / supported by a set of facts. Give a binary score 'yes' or 'no' score to indicate 
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
    single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)


answer_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)

clasify_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant specialized in categorizing documents from the Spanish 
    "Boletín Oficial del Estado" (BOE). Your task is to classify the provided text using the specified list of labels. The posible labels are: {list_labels}
    If the text does not clearly fit any of these labels or requires a more general categorization, assign the label "other".
    Provide the value label as a JSON with a single key 'Label'.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Text: {text} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["text","list_labels"],
)



### CHAINS
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
