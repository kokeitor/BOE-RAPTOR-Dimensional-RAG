import torch.nn as nn
import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data.dataset import ConcatDataset
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import ConcatDataset
import requests
from typing import List, Tuple, Dict, Optional
from datasets import load_dataset, DatasetDict
from transformers.integrations import TensorBoardCallback
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig, EarlyStoppingCallback
from sklearn.metrics import f1_score
from datasets import Dataset as ds
import sys
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from datetime import datetime, timezone
import json 


# MODULE CLASS DOCU:
"""nn.Module :
Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes: 
self.sub_module = nn.Linear(...)"""

# Iterate along all de modules inside a network class or model class. Notice that LinearRegression module has inside a linear layer "module" or only linear layer
# note that the atribute name : self.linear will define the string "linear" to refer to that layer inside a module
# this will be useful when ,inside a class module, there are several layers


### API KEYS"
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

# MODELS
MODEL_NAME = "microsoft/deberta-base"
MODEL_NAME_2 = "PlanTL-GOB-ES/roberta-base-bne"


BERT_TOKENIZER = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
ROBERTA_TOKENIZER = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")

# Embedding model
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Request to create embeddings: hg api
model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {os.getenv('HUG_API_KEY')}"}




class BoeNetVanilla(nn.Module):
    def __init__(self, d_model :int , boe_labels : int) -> None:
       super().__init__()
       self.d_model = d_model # (384)
       self.boe_labels = boe_labels # (?)
       
       # Arquitecture 
       # (batch, 384)
       self.l_1 = nn.Linear(in_features= self.d_model, out_features = 700, bias = True)
       self.tan_h = nn.Tanh()
       self.drop_1 = nn.Dropout(p = 0.2)
       self.l_2 = nn.Linear(in_features= 700, out_features = 1200, bias = True)
       self.Relu = nn.ReLU()
       self.drop_2 = nn.Dropout(p = 0.3)
       # (batch, 10)
       self.l_3 = nn.Linear(in_features=  1200 , out_features = self.boe_labels, bias = True)
       
    def forward(self,x):
        h = self.drop_1(self.tan_h(self.l_1(x)))
        h = self.drop_2(self.Relu(self.l_2(h)))
        return self.l_3(h)
    
    @staticmethod
    def LossFactory():
        return nn.CrossEntropyLoss()
    
    @staticmethod
    def OptimizerFactory(model, lr : float = 0.001 , betas : tuple = (0.9, 0.999), eps: float =1e-08):
        return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)

    
class BoeNet:
    
    def __init__(self,model_conf_path : str):
        self.config_path = model_conf_path
        self.config = self._parse_config()
        self.model_name = self._get_model_name()
        self.model_tokenizer = self._get_tokenizer()
        self.model_config = self._get_model_config()
        self.model = self._get_model()
        self.metrics = self._get_metrics()
        self.train_args = self._get_training_args()

    def _parse_config(self) -> Dict:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        with open(self.config_path, 'r') as file:
            config = json.load(file)
        return config
    
    def _get_model_name(self):
        model = self.config.get("model", {})
        model_name = model.get("name",None)
        if model_name is None:
            raise ValueError("Model name not defined in the model config file")
        return model_name
    
    def _get_model_config(self):
        problem = self.config.get("problem", {})
        self.labels = problem.get("labels_to_pred",None)
        if self.labels is None:
            raise ValueError("Labels to predict not defined in the model config file")
        self.id2label = {i:lab for lab, i in self.labels}
        self.label2id = {lab:i for lab, i in self.labels}
        try:
            return AutoConfig.from_pretrained(
                                                    pretrained_model_name_or_path = self.model_name, 
                                                    num_labels=len(self.labels), 
                                                    id2label=self.id2label
                                            )
        except Exception as e:
            print(f"Error in get Config Model method : {e}")
            
    def _get_model(self):
        try :
            return AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.model_config)
        except Exception as e:
            print(f"Error in get Model method : {e}")

    def _get_tokenizer(self):
        try:
            return AutoTokenizer.from_pretrained(self.model_name)
        except Exception as e:
            print(f"Error in get tokenizer method : {e}")
            
    def _get_metrics(self):
        metrics = self.config.get("metrics", {})
        if metrics is not {}:
            self.opt_metric = metrics["optimize"]
        else:
            self.opt_metric = ""
        return metrics
    
    def _get_training_args(self):
        training_args = self.config.get("training_args", {})
        return TrainingArguments(
                                    training_args.get("dir_path","./MODEL/DEFAULT"), # nombre del modelo (se crea un directorio con este nombre)
                                    evaluation_strategy = training_args.get("evaluation_strategy","epoch"), # evaluamos y guardamos en cada época.
                                    report_to=training_args.get("report_to","tensorboard"), # reportar a tensorboard para poder ver luego el progreso del entrenamiento.
                                    save_strategy = training_args.get("save_strategy","epoch"), # guardamos también en cada época un checkpoint del modelo.
                                    learning_rate= training_args.get("learning_rate",3e-05), # learning rate que usamos en el Adam Optimizer.
                                    per_device_train_batch_size=training_args.get("per_device_train_batch_size",20), # tamaño de batch en entrenamiento.
                                    per_device_eval_batch_size=training_args.get("per_device_eval_batch_size",32), # tamaño de batch en evaluación.
                                    num_train_epochs=training_args.get("num_train_epochs",100), #  número de épocas para entrenar
                                    weight_decay=training_args.get("weight_decay",0.01), # weight decay en el adam optimizer
                                    adam_epsilon=training_args.get("adam_epsilon",1e-08), # valor para el parámetro epsilon de adam.
                                    load_best_model_at_end=training_args.get("load_best_model_at_end",True), # Si queremos cargar o no el mejor modelo (el checkpoint del modelo que mejor rendimiento tiene) al finalizar el entrenamiento, en caso de haber empeorado en algún momento del entrenamiento.
                                    metric_for_best_model=training_args.get("metric_for_best_model",self.opt_metric), # métrica para escoger el mejor modelo.
                                    gradient_accumulation_steps=training_args.get("gradient_accumulation_steps",10), # número de pasos en los que acumular gradiente.
                                    warmup_ratio=training_args.get("warmup_ratio",0.03), # este es el porcentaje de los pasos de entrenamiento que vamos a hacer "warmup", es decir, que vamos a ir subiendo el learning rate desde casi 0 hasta el learning rate escogido.
                                    fp16=training_args.get("fp16",True) # para activar la precisión mixta. NOTE: Si estáis en google colab con T4 como GPU, en lugar de `bf16_True` usa `fp16=True`
                                )
        
    def _get_trainer(self, dataset :Dataset):
        trainer = self.config.get("trainer", {})
        callbacks_args =  trainer.get("callbacks",None)
        early_stopping_args = callbacks_args.get("EarlyStoppingCallback",None)
        try: 
            train_set = dataset["train"]
        except Exception as e:
            print(f"Error in getting train dataset : {e}")
        try:
            val_set = dataset["validation"]
        except Exception as e:
            print(f"Error in getting val dataset : {e}")
            
        return Trainer(
                        model=self.model,
                        args=self.train_args,
                        train_dataset=train_set,
                        eval_dataset=val_set,
                        tokenizer=self._get_tokenizer,
                        compute_metrics=self.compute_metrics,
                        callbacks=[
                                    EarlyStoppingCallback(
                                                                early_stopping_patience = early_stopping_args.get("early_stopping_patience",3), 
                                                                early_stopping_threshold =  early_stopping_args.get("early_stopping_threshold",0.0)
                                                            ), 
                                    TensorBoardCallback
                                    ]
                        )
    def train(self,dataset : Dataset):
        self.trainer = self._get_trainer(dataset)
        self.trainer.train()
    def predict(self,dataset: Dataset):
        try:
            return self.trainer.evaluate(dataset["test"])
        except Exception as e:
            print(f"Error in predict method : {e}")
        

if __name__ == '__main__':
    
    # instanciar capa y crear modelo
    model = BOENet(d_model  = 384 , boe_labels = 10)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Cuda available: ",torch.cuda.is_available())
    model.to(device)
    
    # loss class
    loss = BOENet.LossFactory() 
    
    # crea optimizer para prescendir de actualizacion de pesos a mano
    optimizer = BOENet.OptimizerFactory(model) 
    
    print("\n")
    

    for m in model.modules():
        #print(f"\nModule : {m}")
        pass

    # iterate along all the parameters inside a module ( a module can have a lot of layers with their parameters)
    for p in model.named_parameters(prefix='', recurse=True, remove_duplicate=True):
        #print(f"\nParameter : {p}")
        pass
        
    docs = [
        'hola que tal esto es un ejemplo',
        'hola ejemplo 2'
    ]
    
    embedding = [
        np.random.rand(384),
        np.random.rand(384)
        
    ]
    labels = [
        'label 1',
        'label 2'
    ]
    data = {
        'text':docs ,
        'label': labels,
    }
    df_data = pd.DataFrame(data = data)
    df_data.to_csv(path_or_buf='./Data_BOE/filename.csv')
    print(df_data.head(10))
    
    data = BOEData(path = './Data_BOE/filename.csv')
    print(len(data))
    print((data.x.shape))
    print((data.x[0,:].shape))
    
    BATCH_SIZE = 2
    train_loader = torch.utils.data.DataLoader(dataset = data, batch_size = BATCH_SIZE, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = data, batch_size = BATCH_SIZE, shuffle = False)
    print(data.data["text"])
    print(data.data["label"])
    for i,(x,y)in enumerate(train_loader):
        print(i, x.shape, y.shape)
        print(y)
    
    

    
    