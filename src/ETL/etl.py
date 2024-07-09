import os
import json
import uuid
import tiktoken
import asyncio
import pytz
import pandas as pd
import logging
import logging.config
import logging.handlers
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, DebertaModel, GPT2Tokenizer
from dotenv import load_dotenv
from typing import Dict, List, Union, Optional
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Union, Optional, Callable, ClassVar
import ETL.splitters
import ETL.parsers
import ETL.nlp
import warnings
import matplotlib
from ETL.utils import get_current_spanish_date_iso, setup_logging
from databases.google_sheets import GoogleSheet
from ETL.models import ClassifyChunk
from langchain_nvidia_ai_endpoints import ChatNVIDIA


# Set the default font to DejaVu Sans
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

# Logging configuration
logger = logging.getLogger("ETL_module_logger")  # Child logger [for this module]
# LOG_FILE = os.path.join(os.path.abspath("../../../logs/download"), "download.log")  # If not using json config


# Load environment variables from .env file
load_dotenv()

# Tokenizers
TOKENIZER_GPT3 = tiktoken.encoding_for_model("gpt-3.5")
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
TOKENIZER_LLAMA3 = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer_deberta = AutoTokenizer.from_pretrained("microsoft/deberta-base")


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
            logger.exception(f"ValueError : Unsupported file format: {file_format}")
            raise ValueError(f"Unsupported file format: {file_format}")

    def invoke(self, docs: List[Document]) -> None:

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        name_format =  str(get_current_spanish_date_iso()) + "_" + self.file_name +'.'+ self.file_format
        
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
            template="""You are an assistant specialized in categorizing documents from the SpanishcBoletín Oficial del Estado (BOE).\n
            Your task is to classify the provided text using the specified list of labels. The posible labels are: {labels}\n
            You must provide three posible labels ordered by similarity score with the text content. The similarity scores must be a number between 0 and 1.\n
            Provide the values as a JSON with three keys : 'Label_1','Label_2','Label_3'and for each label two keys : "Label" for the the label name and "Score" the similarity score value.\n
            Text: {text}""",
            input_variables=["text", "labels"]
        )
        self.llama_prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an assistant specialized in categorizing documents from the Spanish Boletín Oficial del Estado (BOE).\n
            Your task is to classify the provided text using the specified list of labels. The posible labels are: {labels}\n
            You must provide three posible labels ordered by similarity score with the text content. The similarity scores must be a number between 0 and 1.\n
            Provide the values as a JSON with three keys : 'Label_1','Label_2','Label_3'and for each label two keys : "Label" for the the label name and "Score" the similarity score value.\n
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Text: {text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            input_variables=["text", "labels"]
        )
        models = {
            'GPT': ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0,model_kwargs={"response_format": {"type": "json_object"}}),
            'NVIDIA-LLAMA3': ChatNVIDIA(model_name='meta/llama3-70b-instruct', temperature=0),
            'LLAMA': ChatOllama(model='llama3', format="json", temperature=0),
            'LLAMA-GRADIENT': ChatOllama(model='llama3-gradient', format="json", temperature=0)
        }

        self.model = models.get(self.model_label, None)
        if self.model is None:
            logger.exception("AttributeError : Model Name not correct")
            raise AttributeError("Model Name not correct")

        if self.model_label == "NVIDIA-LLAMA3":
            self.chain = self.llama_prompt | self.model | JsonOutputParser()
        elif self.model_label == "GPT":
            self.chain = self.prompt | self.model | JsonOutputParser()
            
    def _get_tokens(self, text: str) -> int:
        """Returns the number of tokens in a text string."""
        try :
            enc = tiktoken.get_encoding("cl100k_base")
            num_tokens = len(enc.encode(text))
        except Exception as e:
            num_tokens = len(self.tokenizer(text)["input_ids"])
            logger.exception(f"{e}")
        return num_tokens

    def invoke(self, docs: List[Document]) -> List[Document]:
        
        docs_copy = docs.copy()
        
        for i, doc in enumerate(docs_copy):
            if i >= self.max_samples:
                logger.warning(f"Reached max samples : {self.max_samples} parameter while generating labels")
                break

            chunk_text = doc.page_content
            chunk_tokens = self._get_tokens(text=chunk_text)
            chunk_len = len(chunk_text)

            # Update metadata
            doc.metadata['num_tokens'] = chunk_tokens
            doc.metadata['num_caracteres'] = chunk_len

            # Generate labels
            generation = self.chain.invoke({"text": chunk_text, "labels": self.labels})
            logger.info(f"Generating labels with model : {self.model_label} // using : {self.tokenizer}")
            logger.info(f"Generation by model : {generation}")

            try:
                doc.metadata['label_1_label'] = generation["Label_1"]["Label"]
                doc.metadata['label_1_score'] = generation["Label_1"]["Score"]
                doc.metadata['label_2_label'] = generation["Label_2"]["Label"]
                doc.metadata['label_2_score'] = generation["Label_2"]["Score"]
                doc.metadata['label_3_label'] = generation["Label_3"]["Label"]
                doc.metadata['label_3_score'] = generation["Label_3"]["Score"]
            except Exception as e:
                doc.metadata['label_1_label'] = 'ERROR'
                doc.metadata['label_1_score'] = 0
                doc.metadata['label_2_label'] = 'ERROR'
                doc.metadata['label_2_score'] = 0
                doc.metadata['label_3_label'] = 'ERROR'
                doc.metadata['label_3_score'] = 0
                logger.exception(f"LLM Error message:  : {e}")

        return docs


class Pipeline:
    def __init__(self, config_path: str, database : GoogleSheet):
        self.config_path = config_path
        self.config = self._parse_config()
        self.parser = self._create_parser()
        self.splitter = self._create_splitter()
        self.label_generator = self._create_label_generator()
        self.storer = self._create_storer()
        self.database = database

    def _parse_config(self) -> Dict:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        with open(self.config_path, encoding='utf-8') as file:
            config = json.load(file)
        return config

    def _create_parser(self) -> ETL.parsers.Parser:
        parser_config = self.config.get('parser', {})
        return ETL.parsers.Parser(
            directory_path=os.path.abspath(parser_config.get('directory_path', './data/boe/dias/')),
            file_type=parser_config.get('file_type', '.pdf'),
            recursive_parser=parser_config.get('recursive_parser', True),
            result_type=parser_config.get('result_type', 'markdown'),
            verbose=parser_config.get('verbose', True),
            api_key=parser_config.get('api_key', os.getenv('LLAMA_CLOUD_API_KEY'))
        )

    def _create_processor(self, docs: List[Document]) -> ETL.nlp.BoeProcessor:
        txt_process_config = self.config.get('TextPreprocess', None)
        if txt_process_config is not None:
            spc_words = txt_process_config.get('spc_words', None)
            special_char = txt_process_config.get('spc_caracters', None)
            preprocess_task = txt_process_config.get('task_name', "Default")
            processor = ETL.nlp.BoeProcessor(task=preprocess_task, docs=docs, spc_caracters=special_char, spc_words=spc_words)
            
            txt_process_methods = txt_process_config.get('methods', None)
            logger.info(f"Configuration of TextPreprocess for task : {preprocess_task} found")
            
            for method_key, method_vals in txt_process_methods.items():
                if method_vals.get("apply", False):
                    logger.info(f"Trying to preprocess texts --> {method_key} : {method_vals}")
                    if method_key == 'del_stopwords':
                        processor = processor.del_stopwords(lang=method_vals.get("lang", "Spanish"))
                    elif method_key == 'del_urls':
                        processor = processor.del_urls()
                    elif method_key == 'del_html':
                        processor = processor.del_html()
                    elif method_key == 'del_emojis':
                        processor = processor.del_emojis()
                    elif method_key == 'del_special':
                        processor = processor.del_special()
                    elif method_key == 'del_digits':
                        processor = processor.del_digits()
                    elif method_key == 'del_special_words':
                        processor = processor.del_special_words()
                    elif method_key == 'del_chinese_japanese':
                        processor = processor.del_chinese_japanese()
                    elif method_key == 'del_extra_spaces':
                        processor = processor.del_extra_spaces()
                    elif method_key == 'get_lower':
                        processor = processor.get_lower()
                    elif method_key == 'get_alfanumeric':
                        processor = processor.get_alfanumeric()
                    elif method_key == 'stem':
                        processor = processor.stem()
                    elif method_key == 'lemmatizer':
                        processor = processor.lemmatize()
                    elif method_key == 'custom_del':
                        path = os.path.abspath(method_vals.get("storage_path","./data/figures/text/process"))
                        abs_path_name = os.path.join(path, f"{get_current_spanish_date_iso()}.png")
                        logger.info(f"Path to save plot 'custom_del' : {abs_path_name}")
                        _, _ = processor.custom_del(
                            text_field_name="text",
                            data=self.get_dataframe(docs=docs),
                            delete=method_vals.get("delete", False),
                            plot=method_vals.get("plot", True),
                            storage_path=abs_path_name
                        )
                    elif method_key == 'bow':
                        path = os.path.abspath(method_vals.get("storage_path","./data/figures/text/bow"))
                        abs_path_name = os.path.join(path, f"{get_current_spanish_date_iso()}.png")
                        logger.info(f"Path to save plot 'bow' : {abs_path_name}")
                        self.save_figure_from_df(
                            df=processor.bow(),
                            path=abs_path_name,
                            method='BOW'
                        )
                    elif method_key == 'bow_tf_idf':
                        path = os.path.abspath(method_vals.get("storage_path", "./data/figures/text/bow"))
                        abs_path_name = os.path.join(path, f"{get_current_spanish_date_iso()}.png")
                        logger.info(f"Path to save plot 'bow_tf_idf' : {abs_path_name}")
                        self.save_figure_from_df(
                            df=processor.bow_tf_idf(),
                            path=abs_path_name,
                            method='BOW-TF-IDF'
                        )
                    else:
                        logger.warning(f"Method {method_key} not found for TextPreprocess class")
        else:
            logger.warning("Configuration of TextPreprocess not found, applying default one")
            preprocess_task = "Default process config"
            processor = ETL.nlp.BoeProcessor(task=preprocess_task, docs=docs)

        return processor


    def _create_splitter(self) -> ETL.splitters.Splitter:
        splitter_config = self.config.get('splitter', {})
        return ETL.splitters.Splitter(
            chunk_size=splitter_config.get('chunk_size', 200),
            embedding_model=self._get_embd_model(embd_model=splitter_config.get('embedding_model', str(os.getenv('EMBEDDING_MODEL')))),
            tokenizer_model=self._get_tokenizer(tokenizer_model=splitter_config.get('tokenizer_model', 'LLAMA3')),
            threshold=splitter_config.get('threshold', 75),
            max_tokens=splitter_config.get('max_tokens', 500),
            verbose=splitter_config.get('verbose', 0),
            buffer_size=splitter_config.get('buffer_size', 3),
            max_big_chunks=splitter_config.get('max_big_chunks', 4),
            splitter_mode=splitter_config.get('splitter_mode', 'CUSTOM'),
            storage_path=os.path.abspath(splitter_config.get('storage_path', "./data/figures/splitter")),
            min_initial_chunk_len=splitter_config.get('min_initial_chunk_len',50)
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
            store_path=os.path.abspath(storer_config.get('store_path', './data/boedataset')),
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
            str(os.getenv('EMBEDDING_MODEL')): HuggingFaceEmbeddings(model_name=str(os.getenv('EMBEDDING_MODEL')))
        }
        return embd_available.get(embd_model, HuggingFaceEmbeddings(model_name=str(os.getenv('EMBEDDING_MODEL'))))

    def get_dataframe(self,docs :List[Document]) -> pd.DataFrame:
        texts = [d.page_content for d in docs]
        return pd.DataFrame(data=texts, columns=["text"])
    
    def save_figure_from_df(self, df: pd.DataFrame, path: str, method: str) -> None:
        most_frequent_tokens = df.sum(axis=0, skipna=True).sort_values(ascending=False)
        num_tokens = 50
        fig = plt.figure(figsize=(16, 10))
        plt.bar(x=most_frequent_tokens.head(num_tokens).index, height=most_frequent_tokens.head(num_tokens).values)
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Most frequent {num_tokens} tokens/terms in corpus using {method} method")
        plt.grid()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, format='png')
        plt.close(fig)

    def run(self) -> List[Document]:
        self.parsed_docs = self.parser.invoke()
        self.processor = self._create_processor(docs=self.parsed_docs)
        processed_docs = self.processor.invoke()
        logger.info(f"Number of processed_docs {len(processed_docs)}")
        try:
            logger.debug(f"Type of processed_docs[0] {type(processed_docs[0])}")
        except:
            pass
        split_docs = self.splitter.invoke(processed_docs)
        labeled_docs = self.label_generator.invoke(split_docs)
        self.storer.invoke(labeled_docs)
        # Saving in bbdd [google sheets]
        for doc in labeled_docs:
            record = {"text": doc.page_content}
            record.update(doc.metadata)
            logger.info(f"Prev the insertion in BBDD -> {record}")
            chunk = ClassifyChunk(**record)
            self.database.write_data(
                            range=self.database.get_last_row_range(), 
                            values=[GoogleSheet.get_record(chunk=chunk)]
                            )
        logger.info(f"Inserting into BBDD -> {chunk}")
        return labeled_docs
    
        