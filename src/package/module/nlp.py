import os
import json
import re
import uuid
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from typing import Dict, List, Union, Optional
from langchain.schema import Document
from langchain_community.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Union, Optional, Callable, ClassVar
from dataclasses import dataclass, field
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from dataclasses import dataclass, Field
import matplotlib.pyplot as plt
import seaborn as sns
import logging


# NLP
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')


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


# Logging configuration
logger = logging.getLogger("nlp_module_logger")  # Child logger [for this module]
# LOG_FILE = os.path.join(os.path.abspath("../../../logs/download"), "download.log")  # If not using json config


#util functions
def get_current_utc_date_iso():
    # Get the current date and time in UTC and format it directly
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


@dataclass
class TextPreprocess:
    """Class for text preprocess"""
    SPC_CARACTERS : ClassVar =  [
                                    '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '=', '+',
                                    '{', '}', '[', ']', '|', '\\', ':', ';', '"', "'", '<', '>', ',', '.',
                                    '?', '/', '~', '`', '\n', '\r', '\t', '\b', '\f','__'
                                ]
    PATRON_EMOJI  : ClassVar = re.compile(
                                            "["
                                            "\U0001F600-\U0001F64F"  # Emoticons
                                            "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
                                            "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
                                            "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
                                            "\U00002700-\U000027BF"  # Dingbats
                                            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                            "\U00002600-\U000026FF"  # Miscellaneous Symbols
                                            "\U00002B50-\U00002B55"  # Additional symbols
                                            "\U00002300-\U000023FF"  # Miscellaneous Technical
                                            "\U0000200D"             # Zero Width Joiner
                                            "\U00002500-\U000025FF"  # Geometric Shapes
                                            "\U00002100-\U0000219F"  # Arrows
                                            "]+",
                                            flags=re.UNICODE,
                                        )
    
    PATRON_CH_JAP: ClassVar = re.compile(
                                    r'[\u4e00-\u9fff]|'  # Basic Chinese
                                    r'[\u3400-\u4dbf]|'  # Extended Chinese
                                    r'[\u3040-\u309f]|'  # Hiragana
                                    r'[\u30a0-\u30ff]|'  # Katakana
                                    r'[\uff66-\uff9f]'   # Half-width Katakana
                                    )
    task : str
    docs : List[Document]
    spc_caracters : Optional[List[str]] = field(default_factory=list)
    data : Optional[pd.DataFrame] = None
    
    def __post__init(self):
        self.corpus = [d.page_content for d in self.docs]
    
    def del_stopwords(self, lang: str) -> 'TextPreprocess':
        empty_words = set(stopwords.words(lang))
        for i, t in enumerate(self.corpus):
            self.corpus[i] = ' '.join([word for word in t.split() if word.lower() not in empty_words])
        return self

    def del_urls(self) -> 'TextPreprocess':
        patron_url = re.compile(r'https?://\S+|www\.\S+')
        for i, t in enumerate(self.corpus):
            processed_text = re.sub(patron_url, '', t)
            self.corpus[i] = re.sub(r'\s+', ' ', processed_text.strip())
        return self

    def del_html(self) -> 'TextPreprocess':
        html_tags_pattern = re.compile(r'<.*?>')
        for i, t in enumerate(self.corpus):
            processed_text = re.sub(html_tags_pattern, '', t)
            self.corpus[i] = re.sub(r'\s+', ' ', processed_text.strip())
        return self

    def del_emojis(self) -> 'TextPreprocess':
        for i, t in enumerate(self.corpus):
            processed_text = re.sub(TextPreprocess.PATRON_EMOJI, '', t)
            self.corpus[i] = re.sub(r'\s+', ' ', processed_text.strip())
        return self
    
    def del_special(self) -> 'TextPreprocess':
        for i, t in enumerate(self.corpus):
            self.corpus[i] = ''.join([c for c in t if c != self.spc_caracters])
        return self

    def del_digits(self) -> 'TextPreprocess':
        for i, t in enumerate(self.corpus):
            processed_text = re.sub(r'[0-9]+', '', t)
            self.corpus[i] = re.sub(r'\s+', ' ', processed_text.strip())
        return self
    
    def del_chinese_japanese(self) -> 'TextPreprocess':
        for i, t in enumerate(self.corpus):
            processed_text = re.sub(TextPreprocess.PATRON_CH_JAP, '', t)
            self.corpus[i] = re.sub(r'\s+', ' ', processed_text.strip())
        return self
    
    def del_extra_spaces(self) -> 'TextPreprocess':
        for i, t in enumerate(self.corpus):
            self.corpus[i] = re.sub(r'\s+', ' ', t.strip())
        return self

    def get_lower(self) -> 'TextPreprocess':
        for i, t in enumerate(self.corpus):
            self.corpus[i] = t.lower()
        return self

    def get_alfanumeric(self) -> 'TextPreprocess':
        for i, t in enumerate(self.corpus):
            processed_text = re.sub(r'[^\w\s]|_', '', t)
            self.corpus[i] = re.sub(r'\s+', ' ', processed_text.strip())
        return self

    def stem(self) -> 'TextPreprocess':
        Porter = PorterStemmer()
        for i, t in enumerate(self.corpus):
            word_tokens = t.split()
            stems = [Porter.stem(word) for word in word_tokens]
            self.corpus[i] = ' '.join(stems)
        return self

    def lemmatize(self) -> 'TextPreprocess':
        lemmatizer = WordNetLemmatizer()
        for i, t in enumerate(self.corpus):
            word_tokens = t.split()
            lemmas = [lemmatizer.lemmatize(word, pos='v') for word in word_tokens]
            self.corpus[i] = ' '.join(lemmas)
        return self
    
    def custom_del(
                self,
                text_field_name: str,
                special_c: Optional[List[str]] = None,
                data: Optional[Union[pd.DataFrame,str]] = None,
                delete: bool = False,
                plot: bool = False
                ) -> Tuple[dict, pd.DataFrame]: 
        
        """Method for custom preprocess/delete chaaracters ferom List[texts] or text (string)"""
        if data is None:
            data = self.data
        if special_c is None:
            special_c = TextPreprocess.SPC_CARACTERS
        
        if data is None:
            raise ValueError("Data must be provided either as a class attribute or as a method parameter.")
        
        if isinstance(data, pd.DataFrame):
            data_is_string = False
            df = data.copy().reset_index(drop=True)  # Reset index here
        elif isinstance(data, str):
            data_is_string = True
            text = data
            df = pd.DataFrame()
        else:
            raise ValueError("Unknown non-process-type of 'data' parameter")

        
        special_c_count = {}
        if not text_field_name:
            raise ValueError("text_field_name must be defined")
        
        for char in special_c:
            count = 0
            patron_busqueda = re.compile(re.escape(char))
            if data_is_string:
                match_obj = patron_busqueda.search(text)
                if char in text or match_obj is not None:
                    count += 1
                    if delete:
                        text = ''.join([c for c in text if c != char])
                        #text = re.sub(patron_busqueda, '', text) # alternative method
                special_c_count[char] = count
            else:
                for i in range(df.shape[0]):
                        text = df.loc[i, text_field_name]
                        match_obj = patron_busqueda.search(text)
                        if char in text or match_obj is not None:
                            count += 1
                            if delete:
                                df.loc[i, text_field_name] = ''.join([c for c in text if c != char])
                                #df.loc[i, text_field_name] = re.sub(patron_busqueda, '', text) # alternative method
                        special_c_count[char] = count
                
        
        if plot:
            plt.figure(figsize=(10, 6))
            plt.bar(special_c_count.keys(), special_c_count.values(), color='skyblue')
            plt.xlabel('Special Characters')
            plt.ylabel('Frequency')
            plt.title('Special Characters in Texts')
            plt.xticks(rotation=45)
            plt.grid()
            plt.show()
            
        if data_is_string:
            return special_c_count, text
        else:
            return special_c_count, df
        
    def bow(self) -> pd.DataFrame:
        vectorizador = CountVectorizer(
            input='content',
            encoding='utf-8',
            decode_error='strict',
            strip_accents=None,
            lowercase=True,
            preprocessor=None,
            tokenizer=None,
            stop_words=None,
            token_pattern=r'(?u)\b\w\w+\b',
            ngram_range=(1, 1),
            analyzer='word',
            max_df=1.0,
            min_df=1,
            max_features=None,
            vocabulary=None,
            binary=False
        )

        try:
            X = vectorizador.fit_transform(self.corpus)
        except UnicodeDecodeError as e:
            logger.exception(f"Error: characters not of the given encoding -> {e}")
            return pd.DataFrame()

        nombres_caracteristicas = vectorizador.get_feature_names_out()
        return pd.DataFrame(data=X.toarray(), columns=nombres_caracteristicas, index=self.corpus)

    def bow_tf_idf(self) -> pd.DataFrame:
        tfidf_vectorizador = TfidfVectorizer(
            input='content',
            encoding='utf-8',
            decode_error='strict',
            strip_accents=None,
            lowercase=True,
            preprocessor=None,
            tokenizer=None,
            analyzer='word',
            stop_words=None,
            token_pattern=r'(?u)\b\w\w+\b',
            ngram_range=(1, 1),
            max_df=1.0,
            min_df=1,
            max_features=None,
            vocabulary=None,
            binary=False,
            dtype=np.float64,
            norm='l2',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False
        )

        try:
            X = tfidf_vectorizador.fit_transform(self.corpus)
        except UnicodeDecodeError as e:
            logger.exception(f"Error: characters not of the given encoding -> {e}")
            return pd.DataFrame()

        terms = tfidf_vectorizador.get_feature_names_out()
        return pd.DataFrame(data=X.toarray(), columns=terms, index=self.corpus)

@dataclass()
class BoeProcessor(TextPreprocess):
    """BOE PREPROCESS DOC AND ADD METADATA TO EACH DOC"""

    def invoke(self, docs: Optional[List[Document]] =None) -> List[Document]:

        logger.info(f"NUMERO DE DOCS A ANALIZAR : {len(docs)}\n\n")
        new_metadata = {}
        titulos = {}
        new_docs = []
        if docs is None:
            logger.info(f"Procesando documentos boe en {self.__name__}")
            self.processed_docs = self.docs.copy()
        else:
            logger.warning(f"Los documentos BOE en {self.__name__} han sido cambiados en el metodo invoke")
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

    def _clean_doc(self, doc: Document) -> Tuple[Dict, Document]:
        """
        Clean the document by removing specific patterns and extracting titles.
        
        Args:
            doc (Document): The document to be cleaned.

        Returns:
            Tuple[Dict, Document]: A dictionary of titles and the cleaned document.
        """
        doc_clean = doc.copy()
        doc_text = doc_clean.page_content

        titles = self._extract_titles(doc_text)
        clean_text = self._remove_patterns(doc_text)

        doc_clean.page_content = clean_text
        return titles, doc_clean

    def _extract_titles(self, text: str) -> Dict:
        """
        Extract titles from the document text using predefined patterns.
        
        Args:
            text (str): The document text to extract titles from.

        Returns:
            Dict: A dictionary of extracted titles.
        """
        title_1 = r'^##(?!\#).*$'
        title_2 = r'^###(?!\#).*$'
        title_3 = r'^####(?!\#).*$'

        titles_1 = list(set([re.sub(r'#', '', t).strip() for t in re.findall(title_1, text, re.MULTILINE)]))
        titles_2 = list(set([re.sub(r'#', '', t).strip() for t in re.findall(title_2, text, re.MULTILINE)]))
        titles_3 = list(set([re.sub(r'#', '', t).strip() for t in re.findall(title_3, text, re.MULTILINE)]))

        patterns_to_eliminate_titles = [
            r'I', r'II', r'III',
            r'Núm. \d+ [A-Za-z]+ \d+ de [A-Za-z]+ de \d{4} Sec. [A-Z]+\. Pág\. \d+',
            r'## Núm. \d+ [A-Za-z]+ \d+ de [A-Za-z]+ de \d{4} Sec. [A-Z]+\. Pág\. \d+',
            r'BOLETÍN OFICIAL DEL ESTADO', r'BOLETÍN OFCAL DEL ESTADO', r'ANEXO',
            r'\b([A-Z]|I{1,2})\.', r'. DSPOSCONES GENERALES',
            r'Núm. 92 Lunes 15 de abril de 2024 Sec. . Pág. 41278',
            r'MNSTERO DE ASUNTOS EXTERORES, UNÓN EUROPEA Y COOPERACÓN'
        ]

        titles_1 = self._clean_titles(titles_1, patterns_to_eliminate_titles)
        titles_2 = self._clean_titles(titles_2, patterns_to_eliminate_titles)
        titles_3 = self._clean_titles(titles_3, patterns_to_eliminate_titles)

        return {f"titulo_{i}": t for i, t in enumerate([titles_1, titles_2, titles_3]) if t}

    def _clean_titles(self, titles: List[str], patterns: List[str]) -> List[str]:
        """
        Clean the titles by removing specific patterns.
        
        Args:
            titles (List[str]): The list of titles to be cleaned.
            patterns (List[str]): The patterns to remove from the titles.

        Returns:
            List[str]: The cleaned titles.
        """
        for pattern in patterns:
            titles = [re.sub(pattern, '', t).strip() for t in titles]
            titles = [t for t in titles if t]
        return titles

    def _remove_patterns(self, text: str) -> str:
        """
        Remove specific patterns from the document text.
        
        Args:
            text (str): The document text to clean.

        Returns:
            str: The cleaned text.
        """
        patterns_to_eliminate = [
            r'^##(?!\#).*$',
            r'^###(?!\#).*$',
            r'^####(?!\#).*$',
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

        clean_text = text
        for pattern in patterns_to_eliminate:
            clean_text = re.sub(pattern, '', clean_text, flags=re.MULTILINE).strip()

        erase_words = ['BOLETÍN', 'OFICIAL', 'DEL', 'ESTADO', 'CONSEJO',
                       'GENERAL', 'DEL', 'PODER', 'JUDICIAL', 'cve', 'Núm', 'ISSN:', 'Pág.', 'Sec.', '### Primero.', '### Segundo.']
        words = clean_text.split(" ")
        shortlisted_words = [w for w in words if w not in erase_words]
        clean_text = ' '.join(shortlisted_words)

        return clean_text

    def _get_date_creation_doc(self, doc: Document):
      doc_copy = doc.copy()
      logger.info(f"file_path: {doc_copy.metadata['file_path']}")
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
    
    



"""
Usage: 

# Solo meses enero y diciemnbre
preprocessor_bow_tf_idf_jan_dec = TextPreprocess(
    task='classification task ',
    corpus=df_solo_jan_dec["post_text"].tolist(),
    spc_caracters = [
                        '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '=', '+',
                        '{', '}', '[', ']', '|', '\\', ':', ';', '"', "'", '<', '>', ',', '.',
                        '?', '/', '~', '`', '\n', '\r', '\t', '\b', '\f','__'
                    ]
)
bow_df = preprocessor_bow \
.del_urls() \
.get_lower() \
.del_chinese_japanese() \
.get_alfanumeric() \
.del_digits() \
.del_emojis() \
.del_special() \
.bow()
"""