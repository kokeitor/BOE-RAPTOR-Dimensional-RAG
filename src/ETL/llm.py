from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from langchain_community.embeddings import GPT4AllEmbeddings 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory



### LLM MODElS
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDING_MODEL_GPT4 = GPT4AllEmbeddings(model_name ="all‑MiniLM‑L6‑v2.gguf2.f16.gguf")
LOCAL_LLM = 'llama3'
llm = ChatOllama(model=LOCAL_LLM, format="json", temperature=0)
gen_llm = ChatOllama(model=LOCAL_LLM, temperature=0)

clasify_prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant specialized in categorizing documents from the Spanish 
    "Boletín Oficial del Estado" (BOE). Your task is to classify the provided text using the specified list of labels. The posible labels are: {list_labels}
    If the text does not clearly fit any of these labels or requires a more general categorization, assign the label "other".
    Provide the value label as a JSON with a single key 'Label'.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Text: {text} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["text","list_labels"],
)


### calsifier grader
clasify_chain = clasify_prompt | llm | JsonOutputParser()
