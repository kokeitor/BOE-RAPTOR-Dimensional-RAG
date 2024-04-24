import deepl
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document


DEEPL_KEY = "f21735bc-db92-4957-8a48-9bed66114a42:fx"  
LOCAL_LLM = 'llama3'

translator = deepl.Translator(DEEPL_KEY)

### Translation function
def transalate(text :str, target_lang : str = "ES" , verbose : int = 0, mode : str = "LOCAL_LLM") -> str:
    
    _target_lang = {
                        "EN-GB":"British english",
                        "EN-US":"United States english",
                        "ES":"Spanish"
        
                    }
    
    if mode == "DEEPL":
        result = translator.translate_text(text = text,source_lang = 'ES', target_lang= target_lang)
        if verbose == 1:
            print(f"texto :\n{text}")
            print(f"Traduccion:\n{result.text}")  
        return result.text
    else:
        if mode == "GPT":
            llm_for_trl = ChatOpenAI(model_name='gpt-4', temperature = 0 )
        if mode == "LOCAL_LLM":
            llm_for_trl = ChatOllama(model=LOCAL_LLM, temperature=0)
        
        transalation_prompt = PromptTemplate(
                        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for translation into spanish language. \n
                        Use the target specify language to translate the text to that language. \n
                        The translation must be the most reliable as posible to the spanish text keeping the technicalities and without translating proper names or names of cities or villages. \n
                        Return the a JSON with a single key 'translation' and no premable or explaination.<|eot_id|><|start_header_id|>user<|end_header_id|>
                        Text: {text} 
                        Target language: {target_language} 
                        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                        input_variables=["text", "target_language"]
                                        )
        trl_chain = transalation_prompt | llm_for_trl | JsonOutputParser()
        
        return trl_chain.invoke({"text": text , "target_language": _target_lang.get(target_lang,"Spanish")})["translation"]
    
    

# Post-processing
def format_docs(docs : List[Document]) -> str:
    """Trasnform List[Documents] into str using doc atribute page_content

    Args:
        docs (_type_): _description_

    Returns:
        _type_: _description_
    """
    return "\n\n".join(doc.page_content for doc in docs)
