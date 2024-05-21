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

print(sys.executable)
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


# torch.cuda.empty_cache()

BERT_TOKENIZER = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
ROBERTA_TOKENIZER = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")


# Request to create embeddings
"""
model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {HUG_API_KEY}"}
"""


# Dataset
class BOEData(Dataset):
    def __init__(self, path: str, labels : List[str] , label_field :List[str], text_field : str, tokenizer = None ,f : int = 1 ,  get_embeddings : bool = False ) :
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
        self.tokenizer = tokenizer # tokenizer
        self.path = path # path
        self.tokens = [] # for plotting number of tokens per chunk
        self.len_texts = [] # for plotting number of caracters per chunk

        self.label_field = label_field
        self.text_field = text_field
        # raw data in form of df
        read_types = {

                        "text_id" : np.int64 ,
                        "num_len" : np.int32,
                        "num_tokens" : np.int64 ,
                        "val_text" : str ,
                        "val_label_1" :str ,
                        "val_score_1": np.float64,
                        "val_label_2" : str,
                        "val_score_2" : np.float64,
                        "val_label_3" :str ,
                        "val_score_3": np.float64
                      }
        self.data = pd.read_csv( filepath_or_buffer = self.path, delimiter = ',', dtype = read_types)
        print(f"INFORMACION PREVIA ANTES PROCESADO DEL DATA SET : \n\tDATASET SHAPE ORIGINAL: {self.data.shape}")

        # limpieza dataframe valores NAN
        #print(self.data.isna().sum())
        #print(self.data.isnull().values.sum())
        self.data.dropna(axis=0, inplace=True)
        self.data.reset_index(drop=True, inplace=True) # reset index numeration after drop nulls
        print(f"\n\tDATASET SHAPE [PRIMER BORRADO DE NULL]: {self.data.shape}")
        #print(self.data.isnull().values.sum())


        # Create samples and target codify labels to train net
        self.mapping =  self._map_labels()
        #print(self.data.columns)
        #print(self.data.head(5))

        #limpieza dataframe valores NAN
        #print(self.data.isnull().values.sum())
        self.data.dropna(axis=0, inplace=True)
        self.data.reset_index(drop=True, inplace=True) # reset index numeration after drop nulls
        print(f"\n\tDATASET SHAPE [SEGUNDO BORRADO DE NULL]: {self.data.shape}")
        #print(self.data.isnull().values.sum())

        if isinstance(self.text_field, str):

            # Si se usa metodo embedding y no teokenizer : Text embedding tensor -> dimension : (num_texts, d_model)
            #print(type(self.data.loc[:,self.text_field].to_list()[0]))
            self.texts  = [str(t) for t in self.data.loc[:,self.text_field].to_list()]


            # Si se pone flag a true self.x son embeddings de los textos [chunks] // de loc ontrario self.x son los propios textos tokenizados con tokenizer
            if get_embeddings:
              self.x = torch.tensor(self._get_embeddings(self.texts))
            else:
              self._tokenize_texts()

        else:
            raise ValueError('text_field parameter must be str type')

        # Target tensor -> dimension : (num_texts, unique_labels)
        self.unique_labels = len(self.mapping.keys())
        self.y = torch.zeros(self.data.shape[0], self.unique_labels)

        # Fill target vector for each text with the 3 score similarity
        for text_index,row in self.data.iterrows():
              #print(row.loc["map_val_label_1"],row.loc["map_val_label_2"],row.loc["map_val_label_3"])
              self.y[text_index,int(row.loc["map_val_label_1"]) - 1] = row.loc["val_score_1"] * self.f # aplly importance factor
              self.y[text_index,int(row.loc["map_val_label_2"]) - 1] = row.loc["val_score_2"] * self.f # aplly importance factor
              self.y[text_index,int(row.loc["map_val_label_3"]) - 1] = row.loc["val_score_3"] * self.f # aplly importance factor

        # Softmax and factor of importance
        _soft = nn.Softmax(dim=1)
        self.y_soft = _soft(self.y) # softmax by rows (row cte and iter softmax function through colunns) and aplly importance factor

        # delete column "Unnamed: 0	"
        if "Unnamed: 0	" in self.data.columns:
          self.data.drop(columns = "Unnamed: 0", inplace = True)

    def __getitem__(self, index):
        return self.x[index] ,self.y_soft[index]
    def __len__(self):
        return self.y_soft.shape[0]
    def __repr__(self):
      return f'(num_texts, d_model) : {self.x["input_ids"].shape} // (num_texts , unique_labels) : {self.y_soft.shape}'

    @property
    def df(self):
      return self.data

    def _map_labels(self):

        # Calculo del diccionario para mapear labels -> int_id
        unique_total_labels = []
        if isinstance(self.label_field, list):
            for i,label in enumerate(self.label_field):
              if isinstance(label, str):
                unique_total_labels.extend(self.data[label].unique())
            unique_total_labels = set(unique_total_labels)
            unique_total_mapping = {str(v):int(i) for i,v in enumerate(unique_total_labels) }
            print("TOTAL UNIQUE LABELS : ",len(unique_total_mapping.keys()))
            #print((unique_total_mapping))
            print("UNIQUE LABELS 1 : ",self.data['val_label_1'].nunique() )
            print("UNIQUE LABELS 2 : ",self.data['val_label_2'].nunique() )
            print("UNIQUE LABELS 3 : ", self.data['val_label_3'].nunique() )

            # mapping pandas columnsdataframe
            for i_label, label in enumerate(self.label_field):
                if isinstance(label, str):
                    self.data[f'map_{label}'] = self.data[label].map(unique_total_mapping)
                else:
                    raise ValueError(f'label {label} inside List : label_field,  must be the name of a column in the csv file and str type')

            # mapping dataset object

            return unique_total_mapping
        else:
            raise ValueError('label_field parameter must be List[str] ')


    def _get_embeddings(self,texts):
        #print(f"Text to embed :  {type(texts)} // {texts}")
        response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
        self.embeddings = response.json()
        return response.json()

    def _tokenize_texts(self):
      if isinstance(self.texts , list):
        if self.tokenizer is not None:
            # roberta
            self.tokens_dict = self.tokenizer(self.texts, padding=True, truncation=True,  return_tensors="pt")
            # deberta
            #self.tokens_dict = self.tokenizer(self.texts, padding=True, truncation=True,  return_tensors="pt", max_length = 512)
            self.x = self.tokens_dict
            #self.t_type_id = self.tokens_dict["token_type_ids"]
            #self.t_attention_m = self.tokens_dict["attention_mask"]
        else:
          raise ValueError('No tokenizer passed as argument')

    def _process_dataset(self, dataset) -> dict:

      text = str(dataset["val_text"]) # aseguramos tipo de dato es str
      # tokenizacion
      tokenized = self.tokenizer(text, padding=False, truncation=True)

      # calculo de tokens y caracteres por texto
      self.tokens.append(len(tokenized["input_ids"]))
      self.len_texts.append(len(text))

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
      tokenized["labels"][self.mapping.get(dataset["val_label_1"], None)] = self.f * dataset["val_score_1"]
      tokenized["labels"][self.mapping.get(dataset["val_label_2"], None)] = self.f * dataset["val_score_2"]
      tokenized["labels"][self.mapping.get(dataset["val_label_3"], None)] = self.f * dataset["val_score_3"]
      tokenized["labels"] = torch.softmax(tokenized["labels"], dim = 0)

      return tokenized


    def get_dataset(self, split : bool = False, tokenize :bool = True ):

      # load original dataset from path
      if isinstance(self.path, str):
        if self.path.split('.')[-1] == 'csv':
          new_path = self.path.split("/")
          new_path = '/'.join(new_path[0:len(new_path)-1])
        else:
          new_path = self.path
        try:
          # Carga desde pandas df
          #dataset = ds.from_pandas(self.data)
          #dataset = dataset.remove_columns(["Unnamed: 0"])
          # Carga desde csv
          dataset = load_dataset(new_path).remove_columns(["Unnamed: 0"])
        except Exception as e:
          print(e)

      if split:
        # Validation 10% de train
        original_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
        original_dataset_trainval = original_dataset["train"].train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict(
                                    {
                                        "train": original_dataset["train"],
                                        "validation": original_dataset_trainval["test"],
                                        "test": original_dataset["test"]
                                    }

                                )
      if tokenize:
        dataset_tokenize = dataset.map(self._process_dataset, batched=False, remove_columns=dataset["train"].column_names)
      else:
        dataset_tokenize = None
      return dataset,dataset_tokenize


    def get_plots(self):

      # Calculate the number of tokens for each document
      if self.tokens != [] and self.len_texts != []:

        # Plotting the histogram of token counts
        plt.figure(figsize=(10, 6))
        plt.hist(self.tokens, bins=30, color="blue", edgecolor="black", alpha=0.7)
        plt.title("Histogram of Token Counts")
        plt.xlabel("Token Count")
        plt.ylabel("Frequency")
        plt.grid(axis="y", alpha=0.75)

        # Plotting the histogram of caracters counts
        plt.figure(figsize=(10, 6))
        plt.hist(self.len_texts, bins=30, color="red", edgecolor="black", alpha=0.7)
        plt.title("Histogram of caracters Counts")
        plt.xlabel("Caracter Count")
        plt.ylabel("Frequency")
        plt.grid(axis="y", alpha=0.75)

        # Display the histogram
        plt.show

        def addlabels(x,y):
          for i in range(len(x)):
              plt.text(i, y[i], y[i], ha = 'center')

        # Plotting ordered chunk vs num tokens/len
        SAMPLE_PLOT_SIZE = 100
        NUM_CHUNKS = np.arange(0,len(self.len_texts), dtype = int)[0:SAMPLE_PLOT_SIZE]
        plt.figure(figsize=(10, 6))
        plt.bar(NUM_CHUNKS, self.tokens[0:SAMPLE_PLOT_SIZE], color="blue", alpha=1)
        addlabels(NUM_CHUNKS, self.tokens[0:SAMPLE_PLOT_SIZE])
        plt.title("Token counts per chunk index")
        plt.xlabel("CHUNK index")
        plt.ylabel("Token counts")
        plt.grid(axis="y", alpha=0.75)

        plt.figure(figsize=(10, 6))
        plt.bar(NUM_CHUNKS, self.len_texts[0:SAMPLE_PLOT_SIZE], color="red", alpha=1)
        addlabels(NUM_CHUNKS, self.len_texts[0:SAMPLE_PLOT_SIZE])
        plt.title("Caracter counts per chunk index")
        plt.xlabel("CHUNK index")
        plt.ylabel("Caracter counts")
        plt.grid(axis="y", alpha=0.75)

        # Display the histogram
        plt.show