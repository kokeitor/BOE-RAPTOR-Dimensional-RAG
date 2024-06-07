import os
import json
import re
import uuid
import torch
import tiktoken
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, DebertaModel, GPT2Tokenizer
from dotenv import load_dotenv
from typing import Dict, List, Union, Optional
from langchain.schema import Document
from langchain_community.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import nest_asyncio  # only for Jupyter notebook
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Union, Optional, Callable, ClassVar



# Load environment variables from .env file
load_dotenv()

# Set environment variables
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')
os.environ['LLAMA_CLOUD_API_KEY'] = os.getenv('LLAMA_CLOUD_API_KEY')
os.environ['HF_TOKEN'] = os.getenv('HUG_API_KEY')

# Tokenizers
TOKENIZER_GPT3 = tiktoken.encoding_for_model("gpt-3.5")
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
TOKENIZER_LLAMA3 = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer_deberta = AutoTokenizer.from_pretrained("microsoft/deberta-base")
tokenizer_roberta = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")

# Embedding model
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

#util functions
def get_current_utc_date_iso():
    # Get the current date and time in UTC and format it directly
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


class Parser:
    def __init__(self, directory_path: str, file_type: str = ".pdf", recursive_parser: bool = True, result_type: str = "markdown", verbose: bool = True, api_key: str = os.getenv('LLAMA_CLOUD_API_KEY')):
        self.path = directory_path
        self.parser = LlamaParse(
            api_key=api_key,
            result_type=result_type,  # "markdown" and "text" are available
            verbose=verbose
        )

        self.reader = SimpleDirectoryReader(
            input_dir=self.path,
            file_extractor={file_type: self.parser},
            recursive=recursive_parser,  # recursively search in subdirectories
            required_exts=[file_type]
        )

    async def invoke(self) -> List[Document]:
        self.llama_parsed_docs = await self.reader.aload_data()  # returns List[llama doc objt]
        self.lang_parsed_docs = [d.to_langchain_format() for d in self.llama_parsed_docs]
        return self.lang_parsed_docs



class Storer:
    def __init__(self, store_path: str,file_name :str, file_format : str = 'csv'):
        self.store_path = store_path
        self.file_name = file_name
        self.file_format = file_format.lower()

    def _document_to_dataframe(self, docs: List[Document]) -> pd.DataFrame:
        records = []
        for doc in docs:
            record = {"text": doc.page_content}
            record.update(doc.metadata)
            records.append(record)
        return pd.DataFrame(records)
      
    def _get_id(self) -> str:
      return str(uuid.uuid4())

    def _store_dataframe(self, df: pd.DataFrame, path: str, file_format: str) -> None:
        if file_format == "parquet":
            df.to_parquet(path, index=False)
        elif file_format == "csv":
            df.to_csv(path, index=False)
        elif file_format == "feather":
            df.to_feather(path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    def invoke(self, docs: List[Document]) -> None:

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        name_format =  str(get_current_utc_date_iso()) + "_" + self.file_name +'.'+ self.file_format
        df = self._document_to_dataframe(docs)
        full_path = os.path.join(self.store_path, name_format)
        self._store_dataframe(df,full_path, self.file_format)
        
        
class LabelGenerator:
    LABELS = """Leyes Orgánicas,Reales Decretos y Reales Decretos-Leyes,Tratados y Convenios Internacionales,Leyes de Comunidades Autónomas,Reglamentos y Normativas Generales,
    Nombramientos y Ceses,Promociones y Situaciones Especiales,Convocatorias y Resultados de Oposiciones,Anuncios de Concursos y Adjudicaciones de Plazas,
    Ayudas, Subvenciones y Becas,Convenios Colectivos y Cartas de Servicio,Planes de Estudio y Normativas Educativas,Convenios Internacionales y Medidas Especiales,
    Edictos y Notificaciones Judiciales,Procedimientos y Citaciones Judiciales,Licitaciones y Adjudicaciones Públicas,Avisos y Notificaciones Oficiales,
    Anuncios Comerciales y Convocatorias Privadas,Sentencias y Autos del Tribunal Constitucional,Orden de Publicaciones y Sumarios,Publicaciones por Órgano Emisor,
    Jerarquía y Autenticidad de Normativas,Publicaciones en Lenguas Cooficiales,Interpretaciones y Documentos Oficiales,Informes y Comunicaciones de Interés General,
    Documentos y Estrategias Nacionales,Medidas de Emergencia y Seguridad Nacional,Anuncios de Regulaciones Específicas,Normativas Temporales y Urgentes,
    Medidas y Políticas Sectoriales,Todos los Tipos de Leyes (Nacionales y Autonómicas),Todos los Tipos de Decretos (Legislativos y no Legislativos),
    Convocatorias y Resultados Generales (Empleo y Educación),Anuncios y Avisos (Oficiales y Privados),
    Judicial y Procedimientos Legales,Sentencias y Declaraciones Judiciales,Publicaciones Multilingües y Cooficiales,Informes y Estrategias de Política,
    Emergencias Nacionales y Medidas Excepcionales,Documentos y Comunicaciones Específicas"""

    def __init__(self, tokenizer = TOKENIZER_GPT3, labels: Optional[List[str]] = None, model: str = 'GPT', max_samples: int = 10):
        self.model_label = model
        self.max_samples = max_samples
        self.tokenizer = tokenizer
        if labels is None:
            self.labels = LabelGenerator.LABELS
        else:
            self.labels = labels

        self.prompt = PromptTemplate(
            template="""system You are an assistant specialized in categorizing documents from the Spanish
            Boletín Oficial del Estado (BOE). Your task is to classify the provided text using the specified list of labels. The posible labels are: {labels}
            You must provide three posible labels ordered by similarity score with the text content. The similarity scores must be a number between 0 and 1.
            Provide the values as a JSON with three keys : 'Label_1','Label_2','Label_3'and for each label two keys : "Label" for the the label name and "Score" the similarity score value.
            user
            Text: {text} assistant""",
            input_variables=["text", "labels"]
        )
        models = {
            'GPT': ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0),
            'LLAMA': ChatOllama(model='llama3', format="json", temperature=0),
            'LLAMA-GRADIENT': ChatOllama(model='llama3-gradient', format="json", temperature=0)
        }

        self.model = models.get(self.model_label, None)
        if self.model is None:
            raise AttributeError("Model Name not correct")

        self.chain = self.prompt | self.model | JsonOutputParser()

    def _get_tokens(self, text: str) -> int:
        """Returns the number of tokens in a text string."""
        try :
          enc = tiktoken.get_encoding("cl100k_base")
          num_tokens = len(enc.encode(text))
        except:
          num_tokens = len(self.tokenizer(text)["input_ids"])
        return num_tokens

    def invoke(self, docs: List[Document]) -> List[Document]:
      docs_copy = docs.copy()
      for i, doc in enumerate(docs_copy):
          if i >= self.max_samples:
              break

          chunk_text = doc.page_content
          chunk_tokens = self._get_tokens(text=chunk_text)
          chunk_len = len(chunk_text)

          # Update metadata
          doc.metadata['num_tokens'] = chunk_tokens
          doc.metadata['num_caracteres'] = chunk_len

          # Generate labels
          generation = self.chain.invoke({"text": chunk_text, "labels": self.labels})
          print("genreation :", generation)

          try:
              doc.metadata['label_1_label'] = generation["Label_1"]["Label"]
              doc.metadata['label_1_score'] = generation["Label_1"]["Score"]
              doc.metadata['label_2_label'] = generation["Label_2"]["Label"]
              doc.metadata['label_2_score'] = generation["Label_2"]["Score"]
              doc.metadata['label_3_label'] = generation["Label_3"]["Label"]
              doc.metadata['label_3_score'] = generation["Label_3"]["Score"]
          except Exception as e:
              print("LLM Error message: ", e)
              doc.metadata['label_1_label'] = 'ERROR'
              doc.metadata['label_1_score'] = 0
              doc.metadata['label_2_label'] = 'ERROR'
              doc.metadata['label_2_score'] = 0
              doc.metadata['label_3_label'] = 'ERROR'
              doc.metadata['label_3_score'] = 0

      return docs


class Pipeline:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._parse_config()
        self.parser = self._create_parser()
        self.processor = self._create_processor()
        self.splitter = self._create_splitter()
        self.label_generator = self._create_label_generator()
        self.storer = self._create_storer()

    def _parse_config(self) -> Dict:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        with open(self.config_path, 'r') as file:
            config = json.load(file)
        return config

    def _create_parser(self) -> Parser:
        parser_config = self.config.get('parser', {})
        return Parser(
            directory_path=parser_config.get('directory_path', ''),
            file_type=parser_config.get('file_type', '.pdf'),
            recursive_parser=parser_config.get('recursive_parser', True),
            result_type=parser_config.get('result_type', 'markdown'),
            verbose=parser_config.get('verbose', True),
            api_key=parser_config.get('api_key', os.getenv('LLAMA_CLOUD_API_KEY'))
        )

    def _create_processor(self) -> Processor:
        return Processor()

    def _create_splitter(self) -> Splitter:
        splitter_config = self.config.get('splitter', {})
        return Splitter(
            chunk_size=splitter_config.get('chunk_size', 200),
            embedding_model=self._get_embd_model(embd_model=splitter_config.get('embedding_model', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')),
            tokenizer_model=self._get_tokenizer(tokenizer_model=splitter_config.get('tokenizer_model', 'LLAMA3')),
            threshold=splitter_config.get('threshold', 75),
            max_tokens=splitter_config.get('max_tokens', 500),
            verbose=splitter_config.get('verbose', 0),
            buffer_size=splitter_config.get('buffer_size', 3),
            max_big_chunks=splitter_config.get('max_big_chunks', 4),
            splitter_mode=splitter_config.get('splitter_mode', 'CUSTOM'),
            storage_path=splitter_config.get('storage_path', 'C:\\Users\\Jorge\\Desktop\\MASTER_IA\\TFM\\proyectoCHROMADB\\data\\figures')
        )

    def _create_label_generator(self) -> LabelGenerator:
        label_generator_config = self.config.get('label_generator', {})
        return LabelGenerator(
            tokenizer=self._get_tokenizer(tokenizer_model=label_generator_config.get('tokenizer_model', 'GPT35')),
            labels=label_generator_config.get('labels', LabelGenerator.LABELS),
            model=label_generator_config.get('model', 'GPT'),
            max_samples=label_generator_config.get('max_samples', 10)
        )

    def _create_storer(self) -> Storer:
        storer_config = self.config.get('storer', {})
        return Storer(
            store_path=storer_config.get('store_path', 'C:\\Users\\Jorge\\Desktop\\MASTER_IA\\TFM\\proyectoCHROMADB\\data\\boedataset'),
            file_name=storer_config.get('file_name', 'data'),
            file_format=storer_config.get('file_format', 'csv')
        )

    def _get_tokenizer(self, tokenizer_model: str):
        tokenizers_available = {
            'GPT35': tiktoken.encoding_for_model("gpt-3.5"),
            'GPT2': GPT2Tokenizer.from_pretrained('gpt2'),
            'LLAMA3': AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B"),
            'DEBERTA': AutoTokenizer.from_pretrained("microsoft/deberta-base"),
            'ROBERTA': AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")
        }
        return tokenizers_available.get(tokenizer_model, AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B"))

    def _get_embd_model(self, embd_model: str):
        embd_available = {
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2': HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        }
        return embd_available.get(embd_model, HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))

    async def run(self) -> List[Document]:
        parsed_docs = await self.parser.invoke()
        processed_docs = self.processor.invoke(parsed_docs)
        split_docs = self.splitter.invoke(processed_docs)
        labeled_docs = self.label_generator.invoke(split_docs)
        self.storer.invoke(labeled_docs)
        return labeled_docs
    
    

if __name__ == '__main__':
    import asyncio

    pipeline = Pipeline(config_path='C:\\Users\\Jorge\\Desktop\\MASTER_IA\\TFM\\proyectoCHROMADB\\config\\etl_config.json')
    result = asyncio.run(pipeline.run())



    text = """ En este apartado se valorará, en su caso, el grado reconocido como personal
    funcionario de carrera en otras Administraciones Públicas o en la Sociedad Estatal de
    Correos y Telégrafos, en el Cuerpo o Escala desde el que participa el funcionario o
    funcionaria de carrera, cuando se halle dentro del intervalo de niveles establecido en el
    artículo 71.1 del Real Decreto 364/1995, de 10 de marzo, para el subgrupo de titulación
    en el que se encuentra clasificado el mismo.

    En el supuesto de que el grado reconocido en el ámbito de otras Administraciones
    Públicas o en la Sociedad Estatal de Correos y Telégrafos exceda del máximo
    establecido en la Administración General del Estado, de acuerdo con el artículo 71 del
    Reglamento mencionado en el punto anterior, para el subgrupo de titulación a que
    pertenezca el funcionario o la funcionaria de carrera, deberá valorársele el grado máximo
    correspondiente al intervalo de niveles asignado a su subgrupo de titulación en la
    Administración General del Estado.

    El funcionario o la funcionaria de carrera que considere tener un grado personal
    consolidado, o que pueda ser consolidado durante el periodo de presentación de
    instancias, deberá recabar del órgano o unidad a que se refiere el apartado 1 de la Base
    Quinta, que dicha circunstancia quede expresamente reflejada en el anexo
    correspondiente al certificado de méritos (anexo II)."""

    """
    s = Splitter(
        embedding_model=EMBEDDING_MODEL,
        tokenizer_model=tokenizer_llama3,
        threshold=75,
        max_tokens=500,
        verbose=1,
        buffer_size=3,
        max_big_chunks=4,
        splitter_mode='CUSTOM',
        embedding_for_research='HG',
        score_threshold_for_research=0.82,
    )
    doc = Document(page_content=text, metadata={"hola": '1'})
    split_docs = s.invoke(docs=[doc])

    # Example of usage:
    pipeline = Pipeline(config_path='path_to_config.json')
    result = pipeline.run()
    
    """
