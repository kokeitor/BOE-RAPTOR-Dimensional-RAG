import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data.dataset import ConcatDataset
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
from typing import List, Tuple, Dict, Optional
from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
from transformers.integrations import TensorBoardCallback
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig, EarlyStoppingCallback
from sklearn.metrics import f1_score
from datasets import Dataset as ds
import matplotlib.pyplot as plt
import sys
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
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

#util functions
def get_current_utc_date_iso():
    # Get the current date and time in UTC and format it directly
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


"""
BERT_TOKENIZER = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
ROBERTA_TOKENIZER = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")

# Embedding model
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Request to create embeddings

model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {HUG_API_KEY}"}
"""


# Dataset
class BOEData(Dataset):
    def __init__(
                    self, 
                    path: str, 
                    file_format: str , 
                    labels : List[str] , 
                    id_field_name :List[str],
                    label_field_name :List[str], 
                    score_field_name :List[str], 
                    text_field_name : str, 
                    tokenizer :Optional[str] = None,
                    f : int = 1 ,  
                    get_embeddings : bool = False,
                    embedding_model :str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
                    ) :
        """
        BOE dataset

        Parameters
        ----------
            key word arguments:
            - f : (int) Importance factor. Is the importance you want to give to the similarity score stablished by the LLM for each label given to each chunk of the text
            - ...

        Return
        -------
            None

        """
        super().__init__()

        self.f = f # Importance factor
        self.labels = labels # labels
        if tokenizer is not None:
            try: 
                self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer))
            except Exception as e:
                raise AttributeError('TOKENIZER MODEL ERROR : ', e)
                
        # embedding model
        try:
            self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        except Exception as e:
            print("EMBEDDING MODEL ERROR : ", e)
            print("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 will be used as model")
            self.embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

            
        self.path = path # path
        self.tokens = [] # for plotting number of tokens per chunk
        self.len_texts = [] # for plotting number of characters per chunk
        self.file_format = file_format # files format : parquet, feather, csv

        self.label_names = label_field_name
        self.score_names = score_field_name
        self.text_name = text_field_name
        if id_field_name:
            self.id_field_name = id_field_name
            self.ids = {k:[] for k in id_field_name}
            print('ID FIELDS :',self.ids)
        
        
        self.data = self._get_data(
                            path = self.path, 
                            file_format = self.file_format, 
                            id_field_name = self.id_field_name,
                            label_names = self.label_names ,
                            score_names = self.score_names,
                            text_name = self.text_name 
                            )
        print(f"\nDATASET SHAPE ORIGINAL: {self.data.shape}")
        self.data = self._clean_data(data =self.data)
        
        # delete column "Unnamed: 0"
        if "Unnamed: 0" in self.data.columns:
          self.data.drop(columns = "Unnamed: 0", inplace = True)
        
        # Create samples and target codify labels to train net
        self.mapping =  self._map_labels()
        print('label names mapping : ', self.mapping)

        self.data = self._clean_data(data =self.data)

        # x samples
        self.texts  = [str(t) for t in self.data.loc[:,self.text_name].to_list()]
        self.x = self._get_x_set(text_name = self.text_name, texts = self.texts, data = self.data, get_embeddings=get_embeddings )
        
        # y samples
        self.y, self.y_soft = self._get_y_set(mapping =self.mapping , data =self.data)
    

    def __getitem__(self, index):
        return self.x[index] ,self.y_soft[index]
    def __len__(self):
        return self.y_soft.shape[0]
    def __repr__(self):
      return f'(num_texts, d_model) : {self.x["input_ids"].shape} // (num_texts , unique_labels) : {self.y_soft.shape}'
    
    @property
    def df(self):
        return self.data

  
    def _get_data(self,  path :str, file_format :str ,id_field_name  :List[str] , label_names :List[str] , score_names : List[str],text_name : str) -> pd.DataFrame:
        # Check if the directory exists
        if not os.path.isdir(path):
            raise FileNotFoundError(f"The directory {path} does not exist.")
        
        # Initialize an empty list to store dataframes
        dataframes = []
        
        # columns names list to read (correct type list)
        text_name = text_name if isinstance(text_name, list) else [text_name]
        label_names = label_names if isinstance(label_names, list) else [label_names]
        score_names = score_names if isinstance(score_names, list) else [score_names]
        id_field_name = id_field_name if isinstance(id_field_name, list) else [id_field_name]

        # Concatenate all columns to read
        columns = text_name + label_names + score_names + id_field_name
        print(f"COLUMNAS LEIDAS DE : {path}")

        # Loop through all files in the directory
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            # Check the file format and load the file accordingly
            if file_format == 'csv' and file_name.endswith('.csv'):
                df = pd.read_csv(file_path, usecols=columns)
                dataframes.append(df)
            elif file_format == 'feather' and file_name.endswith('.feather'):
                df = pd.read_feather(file_path, columns=columns)
                dataframes.append(df)
            elif file_format == 'parquet' and file_name.endswith('.parquet'):
                df = pd.read_parquet(file_path, columns=columns)
                dataframes.append(df)
                
        # Concatenate all dataframes
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
        else:
            raise ValueError(f"No files with the format {file_format} found in the directory {path}")
        return combined_df
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        
        clean_data = data.copy()
        
        # limpieza dataframe valores NAN
        self.nan = clean_data.isna().sum()
        #print(self.data.isnull().values.sum())
        
        clean_data.dropna(axis=0, inplace=True)
        clean_data.reset_index(drop=True, inplace=True) # reset index numeration after drop nulls
        print(f"\n\tDATASET SHAPE [BORRADO NULLS]: {clean_data.shape}")
        
        return clean_data


    def _map_labels(self):

        # Calculo del diccionario para mapear labels -> int_id
        unique_total_labels = []

        if isinstance(self.label_names, list):
            for _,label in enumerate(self.label_names):
              if isinstance(label, str):
                print(f'UNIQUE LABELS IN{label} : ',self.data[label].nunique())
                unique_total_labels.extend(self.data[label].unique())
            unique_total_labels = set(unique_total_labels)
            unique_total_mapping = {str(v):int(i) for i,v in enumerate(unique_total_labels) }
            
            print("TOTAL UNIQUE LABELS : ",len(unique_total_mapping.keys()))

            # mapping pandas columns dataframe
            for _, label in enumerate(self.label_names):
                if isinstance(label, str):
                    self.data[f'map_{label}'] = self.data[label].map(unique_total_mapping)
                else:
                    raise ValueError(f'label {label} inside List : label_names,  must be the name of a column file and str type')

            return unique_total_mapping
        else:
            raise ValueError('label_names parameter must be List[str] ')
        
        
    def _get_x_set(self, text_name :str, texts :List[str] , data : pd.DataFrame, get_embeddings: bool) -> torch.Tensor:
        _data = data.copy()
        if isinstance(self.text_name, str):
            # Si se usa metodo embedding y no tokenizer : Text embedding tensor -> dimension : (num_texts, d_model)
            #print(type(self.data.loc[:,self.text_name].to_list()[0]))
            texts  = [str(t) for t in _data.loc[:,text_name].to_list()]

            # Si se pone flag a true self.x son embeddings de los textos [chunks] // de lo contrario self.x son los propios textos tokenizados con tokenizer
            if get_embeddings:
              x = torch.tensor(self._get_embeddings(texts = texts))
              return x
            else:
                if self.tokenizer is not None:
                    x = self._tokenize_texts(texts = texts)
                    return x
                else:
                    raise AttributeError('NO TOKENIZER MODEL DEFINED')
        else:
            raise ValueError('text_name parameter must be str type')
    
    def _get_y_set(self, mapping :Dict[str, int] , data : pd.DataFrame):
        
        data_ = data.copy()
        
        # Target tensor -> dimension : (num_texts, unique_labels)
        unique_labels = len(mapping.keys())
        y = torch.zeros(data_.shape[0], unique_labels)

        # Fill target vector for each text with the 3 score similarity
        for text_index,row in data_.iterrows():
              #print(row.loc["map_label_1"],row.loc["map_label_2"],row.loc["map_label_3"])
              y[text_index,int(row.loc["map_label_1_label"])] = row.loc["label_1_score"] * self.f # apply importance factor
              y[text_index,int(row.loc["map_label_2_label"])] = row.loc["label_2_score"] * self.f # apply importance factor
              y[text_index,int(row.loc["map_label_3_label"])] = row.loc["label_3_score"] * self.f # apply importance factor

        # Softmax and factor of importance
        _soft = nn.Softmax(dim=1)
        y_soft = _soft(y) # softmax by rows (row cte and iter softmax function through columns) and apply importance factor
        return y, y_soft
        

    def _get_embeddings(self, texts: List[str]) -> List[float]:
        return self.embedding_model.embed_documents(texts)

    def _tokenize_texts(self, texts : List[str]):
      if isinstance(texts , list):
        if self.tokenizer is not None:
            x = self.tokenizer(texts, padding=True, truncation=True,  return_tensors="pt")
            return x
        else:
          raise ValueError('No tokenizer passed as argument')
    

    def _process_dataset(self, dataset) -> dict:
        

        text_name_field = self.text_name if isinstance(self.text_name, str) else None
        if text_name_field is None:
            raise AttributeError("Text field name must be str")
        
        text = str(dataset[text_name_field]) # aseguramos tipo de dato es str
        
        # tokenizacion
        tokenized = self.tokenizer(text, padding=False, truncation=True)

        # calculo de tokens , caracteres y ids por texto
        self.tokens.append(len(tokenized["input_ids"]))
        self.len_texts.append(len(text))
        for id_i in self.ids.keys():
            self.ids[id_i].append(dataset[id_i])

        # Adicion de nuevos campos al dataset
        """
        tokenized["map_val_label_1"] = self.mapping.get(dataset["val_label_1"], 999)
        tokenized["map_val_label_2"] = self.mapping.get(dataset["val_label_2"], 999)
        tokenized["map_val_label_3"] = self.mapping.get(dataset["val_label_3"], 999)
        """
        tokenized["labels"]  = torch.zeros(len(self.mapping))

        # only using one label:
        #tokenized["labels"] = self.mapping.get(dataset["val_label_1"], 999)

        # using three labels
        tokenized["labels"][self.mapping.get(dataset["label_1_label"], None)] = self.f * dataset["label_1_score"]
        tokenized["labels"][self.mapping.get(dataset["label_2_label"], None)] = self.f * dataset["label_2_score"]
        tokenized["labels"][self.mapping.get(dataset["label_3_label"], None)] = self.f * dataset["label_3_score"]
        tokenized["labels"] = torch.softmax(tokenized["labels"], dim = 0)

        return tokenized


    def get_hg_dataset(self, split : bool = False, tokenize :bool = True ) -> Tuple[Dataset]:

        # load original dataset from path
        try:
            if split:
                # test 20 % train
                df_train_val, df_test = train_test_split(self.data, test_size=0.2, random_state=42)
                print('\n---------------------------------------------------')
                print("Test df shape : ", df_test.shape)
                # Validation 10% de train
                df_train, df_val = train_test_split(df_train_val, test_size=0.1, random_state=42)
                print("Train df shape : ", df_train.shape)
                print("validation df shape : ", df_val.shape)
                print('---------------------------------------------------\n')
                
            
                dataset = DatasetDict({
                                                    "train": ds.from_pandas(df_train),
                                                    "validation": ds.from_pandas(df_val),
                                                    "test":ds.from_pandas(df_test)
                                                    })
            else:
                dataset = DatasetDict({
                                    "train": ds.from_pandas(self.data)
                                    })
            
            dataset = dataset.remove_columns(["__index_level_0__"])
            print("\nHG DATASET :\n ", dataset)
            #dataset = load_dataset(self.path).remove_columns(["Unnamed: 0"])
        except Exception as e:
            print(e)

       
        if tokenize:
            dataset_tokenize = dataset.map(self._process_dataset, batched=False, remove_columns=self.data.columns.tolist())
        else:
            dataset_tokenize = None
        print("\nHG DATASET TOKENIZE:\n ", dataset_tokenize)
        return dataset,dataset_tokenize




    def get_plots(self, dir_path: str, figure_name: str):
        
        def addlabels(x, y, text, size, rotation):
            colors = ['g', 'r', 'c', 'm', 'y', 'k']  # Removed 'b' (blue) from the list
            c = 0 
            for i in range(len(x)):
                if c >= len(colors):
                    c = 0
                plt.annotate(text[i], (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center', size=size, color=colors[c], rotation=rotation)
                c += 1

        # Calculate the number of tokens for each document
        if self.tokens and self.len_texts:
            
            # Plotting the histogram of token counts
            plt.figure(figsize=(10, 6))
            plt.hist(self.tokens, bins=30, color="blue", edgecolor="black", alpha=0.7)
            plt.title("Histogram of Token Counts")
            plt.xlabel("Token Count")
            plt.ylabel("Frequency")
            plt.grid(axis="y", alpha=0.75)
            plot_file = os.path.join(dir_path, get_current_utc_date_iso() + "_" + figure_name + "_" + "token_hist.png")
            plt.savefig(plot_file)
            plt.close()

            # Plotting the histogram of characters counts
            plt.figure(figsize=(10, 6))
            plt.hist(self.len_texts, bins=30, color="red", edgecolor="black", alpha=0.7)
            plt.title("Histogram of Character Counts")
            plt.xlabel("Character Count")
            plt.ylabel("Frequency")
            plt.grid(axis="y", alpha=0.75)
            plot_file = os.path.join(dir_path, get_current_utc_date_iso() + "_" + figure_name + "_" + "character_hist.png")
            plt.savefig(plot_file)
            plt.close()

            # Plotting ordered chunk vs num tokens
            SAMPLE_PLOT_SIZE = min(100, len(self.len_texts))
            NUM_CHUNKS = np.arange(SAMPLE_PLOT_SIZE)
            
            plt.figure(figsize=(10, 6))
            plt.bar(NUM_CHUNKS, self.tokens[:SAMPLE_PLOT_SIZE], color="blue", alpha=1)
            addlabels(NUM_CHUNKS, self.tokens[:SAMPLE_PLOT_SIZE], [str(t) for t in self.tokens[:SAMPLE_PLOT_SIZE]], 10, 0)
            for i,id_i in enumerate(self.id_field_name):
                plot_id = []
                for v in self.ids[id_i][:SAMPLE_PLOT_SIZE]:
                    plot_id.append('#'+ str(id_i)+':'+str(v))
                addlabels(
                            x = NUM_CHUNKS, 
                            y = np.full(len(self.tokens[:SAMPLE_PLOT_SIZE]), np.max(self.tokens[:SAMPLE_PLOT_SIZE]))*(0.1*(i+1)), 
                            text = plot_id, 
                            size = 6, 
                            rotation = 40)
            plt.title("Token counts per chunk index")
            plt.xlabel("CHUNK index")
            plt.ylabel("Token counts")
            plt.grid(axis="y", alpha=0.75)
            plot_file = os.path.join(dir_path, get_current_utc_date_iso() + "_" + figure_name + "_" + "num_token_per_chunk.png")
            plt.savefig(plot_file)
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.bar(NUM_CHUNKS, self.len_texts[:SAMPLE_PLOT_SIZE], color="red", alpha=1)
            addlabels(NUM_CHUNKS, self.len_texts[:SAMPLE_PLOT_SIZE], [str(t) for t in self.len_texts[:SAMPLE_PLOT_SIZE]], 10, 0)
            for i,id_i in enumerate(self.id_field_name):
                plot_id = []
                for v in self.ids[id_i][:SAMPLE_PLOT_SIZE]:
                    plot_id.append('#'+ str(id_i)+':'+str(v))
                addlabels(
                            x = NUM_CHUNKS, 
                            y = np.full(len(self.len_texts[:SAMPLE_PLOT_SIZE]), np.max(self.tokens[:SAMPLE_PLOT_SIZE]))*(0.2*(i+1)), 
                            text = plot_id, 
                            size = 6, 
                            rotation = 40)
            plt.title("Character counts per chunk index")
            plt.xlabel("CHUNK index")
            plt.ylabel("Character counts")
            plt.grid(axis="y", alpha=0.75)
            plot_file = os.path.join(dir_path, get_current_utc_date_iso() + "_" + figure_name + "_" + "num_character_per_chunk.png")
            plt.savefig(plot_file)
            plt.close()



if __name__ =='__main__':
    import os
    import pandas as pd
    import torch
    from transformers import AutoTokenizer
    import re

    # Generate sample data
    def create_sample_data(path):
        data = {
            "text": [
                "This is the first document. Holalaoallaoalaoa ",
                "This document is the second document.hrwhrhyrtrty",
                "And this is the third one.tytrytryrty",
                "Is this the first document?ttrwytrwytry"
            ],
            "label_1_label": ["A", "B", "A", "C"],
            "label_1_score": [0.9, 0.85, 0.8, 0.75],
            "label_2_label": ["B", "C", "B", "A"],
            "label_2_score": [0.1, 0.15, 0.2, 0.25],
            "label_3_label": ["C", "A", "C", "B"],
            "label_3_score": [0.0, 0.05, 0.1, 0.0],
            "pdf_id": [0, 0, 0, 0],
            "chunk_id":  [0, 1, 2, 3]
        }
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(path, "sample_data.csv"), index=False)

    # Ensure the data directory exists
    data_dir = "data/boedataset"
    os.makedirs(data_dir, exist_ok=True)

    # Create sample data
    create_sample_data(data_dir)

    # Test the BOEData class
    def test_BOEData():
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
        labels = ["A", "B", "C"]
        labels = [re.sub("\n", '', l).strip() for l in LABELS.split(',')]
        id_field_name = ['pdf_id','chunk_id']
        label_field_name = ["label_1_label", "label_2_label", "label_3_label"]
        score_field_name = ["label_1_score","label_2_score","label_3_score"]
        text_field_name = "text"
        print(labels)
        print(len(labels))

        boe_data = BOEData(
            path=data_dir,
            file_format="parquet",
            labels=labels,
            id_field_name = id_field_name,
            label_field_name=label_field_name,
            score_field_name = score_field_name,
            text_field_name=text_field_name,
            tokenizer="google-bert/bert-base-cased",
            f=1,
            get_embeddings=False,
            embedding_model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        )

        
        print("Dataset shape:", boe_data.df.shape)
        print("Sample data:")
        print(boe_data.df.head())

        
        dataset, dataset_tokenize= boe_data.get_hg_dataset(split = True, tokenize  = True)

        
        # Generate plots
        boe_data.get_plots(dir_path="C:\\Users\\Jorge\\Desktop\\MASTER_IA\\TFM\\proyectoCHROMADB\\data\\figures", figure_name="prueba")
    
    # Run the test
    test_BOEData()
