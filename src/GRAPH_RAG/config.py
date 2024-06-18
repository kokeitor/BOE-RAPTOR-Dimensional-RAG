import os
import json
import logging
from langchain.prompts import PromptTemplate
from dataclasses import dataclass
from typing import Union, Optional, Callable, ClassVar
from langchain.chains.llm import LLMChain
from pydantic import BaseModel, ValidationError
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from chains import get_chain
from prompts import (
    grader_docs_prompt,
    gen_prompt,
    query_process_prompt,
    hallucination_prompt,
    grade_answer_prompt
    )
from base_models import (
    Analisis,
    Candidato,
    State,
    Agent
)
from graph_utils import (
                        get_current_spanish_date_iso, 
                        get_id
                        )
from exceptions.exceptions import NoOpenAIToken, JsonlFormatError, ConfigurationFileError
from models import (
    get_nvdia,
    get_ollama,
    get_open_ai_json,
    get_open_ai
)

# Logging configuration
logger = logging.getLogger(__name__)

@dataclass()
class ConfigGraph:
    MODEL : ClassVar = {
            "OPENAI": get_open_ai_json,
            "NVIDIA": get_nvdia,
            "OLLAMA": get_ollama
            }

    AGENTS: ClassVar = {
        "docs_grader": Agent(agent_name="docs_grader", model="NVIDIA", get_model=get_nvdia, temperature=0.0, prompt=grader_docs_prompt,parser=JsonOutputParser),
        "query_processor": Agent(agent_name="query_processor", model="NVIDIA", get_model=get_nvdia, temperature=0.0, prompt=query_process_prompt,parser=JsonOutputParser),
        "generator": Agent(agent_name="generator", model="NVIDIA", get_model=get_nvdia, temperature=0.0, prompt=gen_prompt,parser=StrOutputParser),
        "hallucination_grader": Agent(agent_name="hallucination_grader", model="NVIDIA", get_model=get_nvdia, temperature=0.0, prompt=hallucination_prompt,parser=JsonOutputParser),
        "answer_grader": Agent(agent_name="answer_grader", model="NVIDIA", get_model=get_nvdia, temperature=0.0, prompt=grade_answer_prompt,parser=JsonOutputParser),
    }
    config_path: Optional[str] = None
    data_path: Optional[str] = None
    
    def __post_init__(self):
        if self.config_path is None:
            logger.exception("No se ha proporcionado ninguna configuración para la generación usando Agents")
            raise AttributeError("No se ha proporcionado ninguna configuración para la generación usando Agents")
        if self.data_path is None:
            logger.exception("No se han proporcionado datos para analizar para la generación usando Agents")
            raise AttributeError("No se han proporcionado datos para analizar para la generación usando Agents")
        if self.config_path is not None:
            self.config = self.get_config()
            logger.info(f"Definida configuracion mediante archivo JSON en {self.config_path}")
        if self.config_path is not None:
            self.data = self.get_data()
            logger.info(f"Definidos los datos mediante archivo JSON en {self.data_path}")
        
        if len(self.data) > 0:
            self.candidatos = [self.get_candidato(cv=candidato.get("cv", None), oferta=candidato.get("oferta", None)) for candidato in self.data]
        else:
            logger.exception("No se han proporcionado candidatos en el archivo JSON con el correcto fomato [ [cv : '...', oferta : '...'] , [...] ] ")
            raise JsonlFormatError()
        
        # Graph Agents configuration
        self.agents_config = self.config.get("agents", None)
        if self.agents_config is not None:
            self.agents = self.get_agents()

        # Graph configuration
        self.iteraciones = self.config.get("iteraciones", len(self.candidatos))
        self.thread_id = self.config.get("thread_id", "4")
        self.verbose = self.config.get("verbose", 0)
        
    def get_config(self) -> dict:
        if not os.path.exists(self.config_path):
            logger.exception(f"Archivo de configuración no encontrado en {self.config_path}")
            raise FileNotFoundError(f"Archivo de configuración no encontrado en {self.config_path}")
        with open(self.config_path, encoding='utf-8') as file:
            config = json.load(file)
        return config
    
    def get_data(self) -> list[dict[str,str]]:
        if not os.path.exists(self.data_path):
            logger.exception(f"Archivo de configuración no encontrado en {self.data_path}")
            raise FileNotFoundError(f"Archivo de configuración no encontrado en {self.data_path}")
        with open(file=self.data_path, mode='r', encoding='utf-8') as file:
            logger.info(f"Leyendo candidatos en archivo : {self.data_path} : ")
            try:
                data = json.load(file)
            except Exception as e:
                logger.exception(f"Error decoding JSON : {e}")
        return data
    
    def get_agents(self) -> dict[str,Agent]:
        agents = ConfigGraph.AGENTS.copy()
        for agent_graph in agents.keys():
            for agent, agent_config in self.agents_config.items():
                if agent_graph == agent:
                    model = agent_config.get("name", None)
                    model_temperature = agent_config.get("temperature", 0.0)
                    if model is not None:
                        get_model = ConfigGraph.MODEL.get(model, None)
                        if get_model is None:
                            logger.error(f"The Model defined for agemt : {agent} isnt't available -> using OpenAI model")
                            get_model = get_open_ai
                            prompt = self.get_model_agent_prompt(model ='OPENAI', agent = agent)
                        else:
                            prompt = self.get_model_agent_prompt(model = model, agent = agent)
                    else:
                        get_model = get_open_ai(temperature=model_temperature)

                    agents[agent] = Agent(
                        agent_name=agent,
                        model=model,
                        get_model=get_model,
                        temperature=model_temperature,
                        prompt=prompt
                    )
                else:
                    pass
        logger.info(f"Graph Agents : {agents}")
        return agents

    def get_model_agent_prompt(self, model : str, agent : str) -> Union[PromptTemplate,str]:
        if model == 'OPENAI':
            if agent == "docs_grader":
                return ''
            elif agent == "query_processor":
                return ''            
            elif agent == "generator":
                return ''        
            elif agent == "hallucination_grader":
                return ''
            elif agent == "answer_grader":
                return ''
            else:
                logger.exception(f"Error inside confiuration graph file -> Agent with name {agent} does not exist in the graph")
                raise ConfigurationFileError(f"Error inside confiuration graph file -> Agent with name {agent} does not exist in the graph")      
        elif model == 'NVIDIA' or  model =='OLLAMA':
            if agent == "docs_grader":
                return grader_docs_prompt
            elif agent == "query_processor":
                return query_process_prompt        
            elif agent == "generator":
                return gen_prompt      
            elif agent == "hallucination_grader":
                return hallucination_prompt
            elif agent == "answer_grader":
                return grade_answer_prompt
            else:
                logger.exception(f"Error inside confiuration graph file -> Agent with name {agent} does not exist in the graph")
                raise ConfigurationFileError(f"Error inside confiuration graph file -> Agent with name {agent} does not exist in the graph")
        else:
            logger.exception(f"Error inside confiuration graph file -> Model {model} not supported")
            raise ConfigurationFileError(f"Error inside confiuration graph file -> Model {model} not supported")

    def get_candidato(self, cv :str , oferta :str) -> Candidato:
        return Candidato(id=get_id(), cv=cv, oferta=oferta)