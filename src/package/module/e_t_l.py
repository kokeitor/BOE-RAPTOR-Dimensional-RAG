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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import nest_asyncio  # only for Jupyter notebook
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from datetime import datetime, timezone

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


class Processor:
    def __init__(self):
        """PREPROCESS AND ADD METADATA TO EACH DOC"""

    def invoke(self, docs: List[Document]) -> List[Document]:

        print(f"NUMERO DE DOCS A ANALIZAR : {len(docs)}\n\n")
        new_metadata = {}
        titulos = {}
        new_docs = []
        self.processed_docs = docs.copy()
        for _, d in enumerate(self.processed_docs):
            new_metadata["fecha_publicacion_boe"], new_doc = self._get_date_creation_doc(doc=d)
            titulos,new_doc = self._clean_doc(doc=new_doc)
            for k, t in titulos.items():
                new_metadata[k] = t
            new_metadata["pdf_id"] = self._get_id()  # adicion de identificador unico del pdf del que se extrajo dicho doc
            new_docs.append(self._put_metadata(doc=new_doc, new_metadata=new_metadata))
        return new_docs

    def _get_id(self):
        """generate an unique random id and convert it to str"""
        return str(uuid.uuid4())

    def _clean_doc(self, doc: Document) -> Dict:
        doc_clean = doc.copy()
        doc_text = doc_clean.page_content
        title_1 = r'^##(?!\#).*$'
        title_2 = r'^###(?!\#).*$'
        title_3 = r'^####(?!\#).*$'

        patterns_to_elimiate = [
            title_1,
            title_2,
            title_3,
            r'^.*Verificable en https://www\.boe\.es.*$\n?',
            r'BOLETÍN OFICIAL DEL ESTADO',
            r'^.*Núm.*$\n?',
            r'^.*ISSN.*$\n?',
            r'^.*Sec.*$\n?',
            r'^.*cve:*$\n?',
            r'cve: BOE-[A-Z]-\d{4}-\d{4}',
            r'https://www.boe.es',
            r'cve: BOE-[A-Z]-\d{4}-\d{4}',
            r'Núm. \d+ [A-Za-z]+ \d+ de [A-Za-z]+ de \d{4} Sec. [A-Z]+\. Pág\. \d+', 
            r'## Núm. \d+ [A-Za-z]+ \d+ de [A-Za-z]+ de \d{4} Sec. [A-Z]+\. Pág\. \d+', 
            r'BOLETÍN OFICIAL DEL ESTADO',
            r'Lunes \d+ de abril de \d{4}', 
            r'ISSN: \d{4}-\d{3}[XD]'
        ]

        not_include_titles = [r'BOLETÍN OFICIAL DEL ESTADO', r'ANEXO', r'\b([A-Z]|I{1,2})\.']

        titles_1 = list(set([re.sub(r'#', '', t).strip() for t in re.findall(title_1, doc_text, re.MULTILINE)]))
        titles_2 = list(set([re.sub(r'#', '', t).strip() for t in re.findall(title_2, doc_text, re.MULTILINE)]))
        titles_3 = list(set([re.sub(r'#', '', t).strip() for t in re.findall(title_3, doc_text, re.MULTILINE)]))

        clean_text = doc_clean.page_content
        for pattern in patterns_to_elimiate:
            clean_text = re.sub(pattern, '', clean_text, flags=re.MULTILINE).strip()

        errase_words = ['BOLETÍN', 'OFICIAL', 'DEL', 'ESTADO', 'CONSEJO',
                        'GENERAL', 'DEL', 'PODER', 'JUDICIAL', 'cve', 'Núm', 'ISSN:', 'Pág.', 'Sec.', '### Primero.', '### Segundo.']
        words = clean_text.split(" ")
        shortlisted_words = [w for w in words if w not in errase_words]
        clean_text = ' '.join(shortlisted_words)

        doc_clean.page_content = clean_text

        patterns_to_elimiate_tit = [
            r'I',
            r'II',
            r'III',
            r'Núm. \d+ [A-Za-z]+ \d+ de [A-Za-z]+ de \d{4} Sec. [A-Z]+\. Pág\. \d+', 
            r'## Núm. \d+ [A-Za-z]+ \d+ de [A-Za-z]+ de \d{4} Sec. [A-Z]+\. Pág\. \d+', 
            r'BOLETÍN OFICIAL DEL ESTADO',
            r'BOLETÍN OFCAL DEL ESTADO',
            r'ANEXO',
            r'\b([A-Z]|I{1,2})\.',
            r'. DSPOSCONES GENERALES',
            r'Núm. 92 Lunes 15 de abril de 2024 Sec. . Pág. 41278',
            r'MNSTERO DE ASUNTOS EXTERORES, UNÓN EUROPEA Y COOPERACÓN'
        ]

        for pattern in patterns_to_elimiate_tit:
            titles_1 = [re.sub(pattern, '', t).strip() for t in titles_1]
            titles_1 = [item for item in titles_1 if item != '']
            titles_2 = [re.sub(pattern, '', t).strip() for t in titles_2]
            titles_2 = [item for item in titles_2 if item != '']
            titles_3 = [re.sub(pattern, '', t).strip() for t in titles_3]
            titles_3 = [item for item in titles_3 if item != '']

        return {f"titulo_{i}": t for i, t in enumerate([titles_1, titles_2, titles_3]) if t != []} , doc_clean

    def _get_date_creation_doc(self, doc: Document):
      doc_copy = doc.copy()
      print("file_path: ",doc_copy.metadata["file_path"])
      if '/' in doc_copy.metadata["file_path"]:
        dia_publicacion = doc_copy.metadata["file_path"].split("/")[-2]
        mes_publicacion = doc_copy.metadata["file_path"].split("/")[-3]
        año_publicacion = doc_copy.metadata["file_path"].split("/")[-4]
      elif '\\' in doc_copy.metadata["file_path"]:
        dia_publicacion = doc_copy.metadata["file_path"].split("\\")[-2]
        mes_publicacion = doc_copy.metadata["file_path"].split("\\")[-3]
        año_publicacion = doc_copy.metadata["file_path"].split("\\")[-4]
      doc_copy.metadata["fecha_publicacion_boe"] = f"{año_publicacion}-{mes_publicacion}-{dia_publicacion}"
      return f"{año_publicacion}-{mes_publicacion}-{dia_publicacion}", doc_copy

    def _put_metadata(self, doc: Document, new_metadata: Dict) -> None:
        new_doc = doc.copy()
        for key in new_metadata.keys():
            new_doc.metadata[key] = new_metadata.get(key, "Metadata Value not found")
        return new_doc


class CustomSemanticSplitter:
    def __init__(
                 self, 
                 embedding_model = EMBEDDING_MODEL, 
                 tokenizer = TOKENIZER_LLAMA3, 
                 buffer_size: int = 2, 
                 threshold: int = 75, 
                 verbose: int = 0, 
                 max_tokens: int = 500, 
                 max_big_chunks: int = 3,
                 storage_path : str = "C:\\Users\\Jorge\\Desktop\\MASTER_IA\\TFM\\proyectoCHROMADB\\data\\figures"):

        self.buffer_size = buffer_size
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.max_big_chunks = max_big_chunks
        self.embedding_model = embedding_model
        self.verbose = verbose
        self.threshold = threshold
        self.namespace_id = uuid.NAMESPACE_DNS
        self.storage_path = storage_path

    def _prepare_texts(self, doc: Document) -> List[Dict]:
        text = doc.page_content
        self._metadata = doc.metadata.copy()
        # Splitting the text on '.', '?', and '!'
        sentence_list = re.split(r'(?<=[.?!])\s+', text)
        # Split on \n
        #sentence_list = re.split(r'\n', text)
        # Split on \n\n
        #sentence_list = re.split(r'\n\n', text)
        return [{'sentence': d, 'index': i} for i, d in enumerate(sentence_list)]

    def _get_id(self, text: str) -> str:
        return str(uuid.uuid5(self.namespace_id, text))

    def _combine_sentences(self, sentences_to_combine: List[Dict], buffer_size: int = None) -> List[Dict]:
        if buffer_size is None:
            buffer_size = self.buffer_size

        num_sentences_exceed = 0
        sentences = sentences_to_combine.copy()

        for i in range(len(sentences)):
            combined_sentence = ''

            for j in range(i - buffer_size, i):
                if j >= 0:
                    combined_sentence += sentences[j]['sentence'] + ' '

            combined_sentence += sentences[i]['sentence']

            for j in range(i + 1, i + 1 + buffer_size):
                if j < len(sentences):
                    combined_sentence += ' ' + sentences[j]['sentence']

            sentences[i]['combined_sentence'] = combined_sentence

            num_tokens = self._get_tokens(text=combined_sentence)
            if self.verbose == 2:
                print('combined_sentence :', i, '// num_tokens : ', num_tokens)
            if num_tokens > self.max_tokens:
                num_sentences_exceed += 1

        if num_sentences_exceed >= self.max_big_chunks:
            if buffer_size > 1:
                self.buffer_size -= 1
                return self._combine_sentences(sentences_to_combine, buffer_size=self.buffer_size)
            else:
                raise ValueError(f"Buffer size cannot be reduced below 1")

        return sentences

    def _get_embeddings(self, sentences_to_embd: List[str]) -> List[float]:
        return self.embedding_model.embed_documents(sentences_to_embd)

    def _get_similarity(self, embeddings: List[float], similarity: str) -> List[float]:
        metrics = {'COSINE': nn.CosineSimilarity(dim=0, eps=1e-08)}
        embedding_tensors = torch.tensor(embeddings)
        similarity_executer = metrics.get(similarity, None)
        similarity = []
        if similarity_executer is not None:
            for i in range(0, embedding_tensors.shape[0]):
                if i < embedding_tensors.shape[0] - 1:
                    t1 = embedding_tensors[i, :]
                    t2 = embedding_tensors[i + 1, :]
                    similarity.append(1.0 - similarity_executer(t1, t2).item())
                else:
                    similarity.append(similarity[i - 1])
        return similarity

    def _get_chunks(self, sentences: List[dict], threshold: int = 75) -> List[Dict]:
        distances = [x["distance_to_next"] for x in sentences]
        breakpoint_distance_threshold = np.percentile(distances, threshold)
        indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]
        start_index = 0
        chunks = []

        for index in indices_above_thresh:
            end_index = index
            group = sentences[start_index:end_index + 1]
            combined_text = ' '.join([d['sentence'] for d in group])
            chunk_metadata = self._metadata.copy()
            chunk_metadata['chunk_id'] = self._get_id(text=combined_text)
            chunks.append({'chunk_text': combined_text, 'chunk_metadata': chunk_metadata})
            start_index = index + 1

        if start_index < len(sentences):
            combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
            chunk_metadata = self._metadata.copy()
            chunk_metadata['chunk_id'] = self._get_id(text=combined_text)
            chunks.append({'chunk_text': combined_text, 'chunk_metadata': chunk_metadata})
        return chunks

    def _plot_similarity(self, pdf_id : str, sentences: List[Dict], threshold: int = 75):

        distances = [x["distance_to_next"] for x in sentences]
        max_distance = np.max(distances)
        
        plt.figure(figsize=(12, 8))
        plt.plot(distances, marker='o')
        plt.xticks(ticks=np.arange(len(distances)), labels=np.arange(len(distances)))
        y_upper_bound = max_distance * 1.05
        plt.ylim(0, y_upper_bound)
        plt.xlim(0, len(distances))
        breakpoint_distance_threshold = np.percentile(distances, threshold)
        plt.axhline(y=breakpoint_distance_threshold, color='r', linestyle='-')
        num_distances_above_threshold = len([x for x in distances if x > breakpoint_distance_threshold])
        plt.text(x=(len(distances) * .01), y=y_upper_bound / 50, s=f"{num_distances_above_threshold + 1} Chunks")
        indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for i, breakpoint_index in enumerate(indices_above_thresh):
            start_index = 0 if i == 0 else indices_above_thresh[i - 1]
            end_index = breakpoint_index if i < len(indices_above_thresh) - 1 else len(distances)
            plt.axvspan(start_index, end_index, facecolor=colors[i % len(colors)], alpha=0.25)
            plt.text(x=np.average([start_index, end_index]), y=breakpoint_distance_threshold + (y_upper_bound) / 20, s=f"Chunk #{i}", horizontalalignment='center', rotation='vertical')
        if indices_above_thresh:
            last_breakpoint = indices_above_thresh[-1]
            if last_breakpoint < len(distances):
                plt.axvspan(last_breakpoint, len(distances), facecolor=colors[len(indices_above_thresh) % len(colors)], alpha=0.25)
                plt.text(x=np.average([last_breakpoint, len(distances)]), y=breakpoint_distance_threshold + (y_upper_bound) / 20, s=f"Chunk #{i + 1}", rotation='vertical')
        plt.title("Chunks Based On Embedding Breakpoints")
        plt.xlabel("Index sentence")
        plt.ylabel("Similarity distance between pairwise sentences")
        plot_file = os.path.join(self.storage_path, pdf_id+"_"+"similarity_plot.png")
        plt.savefig(plot_file)
        plt.close()

        line = np.arange(0, 10, 0.01)
        plt.figure(figsize=(10, 6))
        plt.hist(distances, alpha=0.5, color='b')
        plt.plot([breakpoint_distance_threshold] * len(line), line, color='r')
        plt.text(x=breakpoint_distance_threshold - 0.01, y=0, s=f" Percentile : {str(threshold)}", rotation='vertical', color='r')
        plt.title("Chunks Based On Embedding Breakpoints")
        plt.ylabel("Similarity distance between pairwise sentences")
        plt.grid(alpha=0.75)
        plot_file_hist = os.path.join(self.storage_path,pdf_id+"_"+"similarity_hist.png")
        plt.savefig(plot_file_hist)
        plt.close()

    def _get_tokens(self, text: str) -> int:
        return len(self.tokenizer(text)["input_ids"])

    def _create_docs(self, chunks: List[Dict]) -> List[Document]:
        return [Document(page_content=chunk_dict['chunk_text'], metadata=chunk_dict['chunk_metadata']) for chunk_dict in chunks]

    def split_documents(self, docs: List[Document]) -> List[Document]:
        self.docs = docs.copy()
        self.spitted_docs = []

        for _, doc in enumerate(self.docs):
            self.doc = doc.copy()
            self.sentences = self._prepare_texts(doc=self.doc)
            self.sentences = self._combine_sentences(sentences_to_combine=self.sentences)
            embed_combined_sentences = self._get_embeddings(sentences_to_embd=[x["combined_sentence"] for x in self.sentences])
            for i, sentence in enumerate(self.sentences):
                sentence["combined_sentence_embedding"] = embed_combined_sentences[i]
            similarities = self._get_similarity(embeddings=embed_combined_sentences, similarity='COSINE')
            for i, sentence in enumerate(self.sentences):
                sentence["distance_to_next"] = similarities[i]
            self.chunks = self._get_chunks(sentences=self.sentences, threshold=self.threshold)
            if self.verbose == 1:
                self._plot_similarity(sentences=self.sentences, threshold=self.threshold, pdf_id = doc.metadata['pdf_id'])
            docs = self._create_docs(chunks=self.chunks)
            self.spitted_docs += docs

        return self.spitted_docs


class Splitter:
    def __init__(self,
                 chunk_size: int = 200,
                 storage_path:str ="C:\\Users\\Jorge\\Desktop\\MASTER_IA\\TFM\\proyectoCHROMADB\\data\\figures", 
                 embedding_model=EMBEDDING_MODEL,
                 tokenizer_model=TOKENIZER_LLAMA3,
                 threshold: int = 75,
                 max_tokens: int = 500,
                 verbose: int = 0,
                 buffer_size: int = 3,
                 max_big_chunks: int = 4,
                 splitter_mode: str = 'CUSTOM'):
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.tokenizer_model = tokenizer_model
        self.threshold = threshold
        self.max_tokens = max_tokens
        self.buffer_size = buffer_size
        self.verbose = verbose
        self.max_big_chunks = max_big_chunks
        self.splitter_mode = splitter_mode
        self.storage_path = storage_path
        self.splitter = self._init_splitter()

    def _init_splitter(self):
        splitter_modes = {
            'RECURSIVE': RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=0,
                length_function=len,
                separators=["\n\n"]
            ),
            'SEMANTIC': SemanticChunker(
                embeddings=self.embedding_model,
                buffer_size=self.buffer_size,
                add_start_index=False,
                breakpoint_threshold_type='percentile',
                breakpoint_threshold_amount=self.threshold,
                number_of_chunks=self.chunk_size,
                sentence_split_regex='(?<=[.?!])\\s+'
            ),
            'CUSTOM': CustomSemanticSplitter(
                embedding_model=self.embedding_model,
                buffer_size=self.buffer_size,
                threshold=self.threshold,
                verbose=self.verbose,
                tokenizer=self.tokenizer_model,
                max_tokens=self.max_tokens,
                max_big_chunks=self.max_big_chunks,
                storage_path=self.storage_path
            )
        }
        return splitter_modes.get(self.splitter_mode)

    def invoke(self, docs: List[Document]) -> List[Document]:
        if isinstance(docs, list):
            return self.splitter.split_documents(docs)
        elif isinstance(docs, Document):
            return self.splitter.split_documents([docs])


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
