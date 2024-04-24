from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from  prompts import routing_prompt,grader_prompt,gen_prompt,hallucination_prompt,answer_prompt

LOCAL_LLM = 'llama3'
llm = ChatOllama(model=LOCAL_LLM, format="json", temperature=0)

### Router chain
router_chain = routing_prompt | llm | JsonOutputParser()

### Grader chain
grader_chain = grader_prompt | llm | JsonOutputParser()

### RAG chain (generation)
rag_chain = gen_prompt | llm | StrOutputParser()

### Hallucination chain (grader)
hallucination_chain = hallucination_prompt | llm | JsonOutputParser()

### Answer grader
answer_chain = answer_prompt | llm | JsonOutputParser()

