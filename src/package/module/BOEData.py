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
        
        self.data = self._get_data(
                            path = self.path, 
                            file_format = self.file_format, 
                            label_names = self.label_names ,
                            score_names = self.score_names,
                            text_name = self.text_name 
                            )
        print(f"INFORMACION PREVIA ANTES PROCESADO DEL DATA SET : \n\tDATASET SHAPE ORIGINAL: {self.data.shape}")
        self.data = self._clean_data(data =self.data)
        # delete column "Unnamed: 0"
        if "Unnamed: 0" in self.data.columns:
          self.data.drop(columns = "Unnamed: 0", inplace = True)
        
        # Create samples and target codify labels to train net
        self.mapping =  self._map_labels()
        print('label names mapping : ', self.mapping)
        #print(self.data.columns)
        #print(self.data.head(5))

        print(f"\n\tDATASET SHAPE ORIGINAL: {self.data.shape}")
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

  
    def _get_data(self,  path :str, file_format :str ,label_names :List[str] , score_names : List[str],text_name : str) -> pd.DataFrame:
        # Check if the directory exists
        if not os.path.isdir(path):
            raise FileNotFoundError(f"The directory {path} does not exist.")
        
        # Initialize an empty list to store dataframes
        dataframes = []
        
        # columns to read
        if isinstance(text_name, list) and isinstance(label_names, list):
            columns= text_name + label_names + score_names
        elif isinstance(text_name, str) and isinstance(label_names, list):
            columns= [text_name] + label_names+ score_names
        elif isinstance(text_name, str) and isinstance(label_names, str):
            columns= [text_name] + [label_names] + [score_names]
             
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
                print(f'UNIQUE LABELS {label} : ',self.data[label].nunique())
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
                    raise AttributeError('NO TOKENIZER MODEL DEFINED : ')
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
            print(x)
            return x
        else:
          raise ValueError('No tokenizer passed as argument')
    

    def _process_dataset(self, dataset) -> dict:

      text = str(dataset["text"]) # aseguramos tipo de dato es str
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
      tokenized["labels"][self.mapping.get(dataset["label_1_label"], None)] = self.f * dataset["label_1_score"]
      tokenized["labels"][self.mapping.get(dataset["label_2_label"], None)] = self.f * dataset["label_2_score"]
      tokenized["labels"][self.mapping.get(dataset["label_3_label"], None)] = self.f * dataset["label_3_score"]
      tokenized["labels"] = torch.softmax(tokenized["labels"], dim = 0)

      return tokenized


    def get_dataset(self, split : bool = False, tokenize :bool = True ):

        # load original dataset from path
        try:
            if split:
                # test 20 % train
                df_train_val, df_test = train_test_split(self.data, test_size=0.2, random_state=42)
                print(type(df_train_val))
                print(df_train_val.shape  ,df_test.shape)
                # Validation 10% de train
                df_train, df_val = train_test_split(df_train_val, test_size=0.1, random_state=42)
                print(df_train.shape ,df_val.shape)
                
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
            print("self.data columns", self.data.columns)
            print("datset : ", dataset)
            #dataset = load_dataset(self.path).remove_columns(["Unnamed: 0"])
        except Exception as e:
            print(e)

       
        if tokenize:
            dataset_tokenize = dataset.map(self._process_dataset, batched=False, remove_columns=self.data.columns.tolist())
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

        # Plotting the histogram of characters counts
        plt.figure(figsize=(10, 6))
        plt.hist(self.len_texts, bins=30, color="red", edgecolor="black", alpha=0.7)
        plt.title("Histogram of Character Counts")
        plt.xlabel("Character Count")
        plt.ylabel("Frequency")
        plt.grid(axis="y", alpha=0.75)

        # Display the histogram
        plt.show()

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
        plt.title("Character counts per chunk index")
        plt.xlabel("CHUNK index")
        plt.ylabel("Character counts")
        plt.grid(axis="y", alpha=0.75)

        # Display the histogram
        plt.show()


if __name__ =='__main__':
    import os
    import pandas as pd
    import torch
    from transformers import AutoTokenizer

    # Generate sample data
    def create_sample_data(path):
        data = {
            "text": [
                "This is the first document.",
                "This document is the second document.",
                "And this is the third one.",
                "Is this the first document?"
            ],
            "label_1_label": ["label1", "label2", "label1", "label3"],
            "label_1_score": [0.9, 0.85, 0.8, 0.75],
            "label_2_label": ["label2", "label3", "label2", "label1"],
            "label_2_score": [0.1, 0.15, 0.2, 0.25],
            "label_3_label": ["label3", "label1", "label3", "label2"],
            "label_3_score": [0.0, 0.05, 0.1, 0.0]
        }
        df = pd.DataFrame(data)
        print("COLUMNAS", df.columns)
        df.to_csv(os.path.join(path, "sample_data.csv"), index=False)

    # Ensure the data directory exists
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    # Create sample data
    create_sample_data(data_dir)

    # Test the BOEData class
    def test_BOEData():
        labels = ["label1", "label2", "label3"]
        label_field_name = ["label_1_label", "label_2_label", "label_3_label"]
        score_field_name = ["label_1_score","label_2_score","label_3_score"]
        text_field_name = "text"
        
        boe_data = BOEData(
            path=data_dir,
            file_format="csv",
            labels=labels,
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

        print("\nData samples (x, y):")
        for i in range(3):
            print(f"Sample {i+1}:")
            print("x:", boe_data[i][0])
            print("y:", boe_data[i][1])
            print()
        
        #
        dataset, dataset_tokenize= boe_data.get_dataset(split = True, tokenize  = True)
        print(dataset)
        print(dataset_tokenize)
        
                # Generate plots
        boe_data.get_plots()
        

    # Run the test
    test_BOEData()
