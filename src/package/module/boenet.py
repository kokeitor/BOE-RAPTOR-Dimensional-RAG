import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig, EarlyStoppingCallback
)
from transformers.integrations import TensorBoardCallback
from transformers import EvalPrediction
from sklearn.metrics import f1_score, recall_score, precision_score
from torcheval.metrics import MulticlassAccuracy
from dotenv import load_dotenv
from typing import Dict, List, Tuple
from datasets import DatasetDict, load_dataset


# MODULE CLASS DOCU:
"""nn.Module :
Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes:
self.sub_module = nn.Linear(...)"""

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

# Util functions
def get_current_utc_date_iso():
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

class BoeNet:
    
    def __init__(self, model_conf_path: str):
        self.config_path = model_conf_path
        self.config = self._parse_config()
        self.model_name = self._get_model_name()
        self.model_tokenizer = self._get_tokenizer()
        self.model_config = self._get_model_config()
        self.model = self._get_model()
        self.compute_metric_f = self._get_metrics()
        self.train_args = self._get_training_args()

    def _parse_config(self) -> Dict:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        with open(self.config_path, 'r') as file:
            config = json.load(file)
        return config
    
    def _get_model_name(self):
        model = self.config.get("model", {})
        model_name = model.get("name", None)
        if model_name is None:
            raise ValueError("Model name not defined in the model config file")
        return model_name
    
    def _get_model_config(self):
        problem = self.config.get("problem", {})
        self.labels = problem.get("labels_to_pred", None)
        if self.labels is None:
            raise ValueError("Labels to predict not defined in the model config file")
        self.id2label = {i: lab for i, lab in enumerate(self.labels)}
        self.label2id = {lab: i for i, lab in enumerate(self.labels)}
        try:
            return AutoConfig.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                num_labels=len(self.labels),
                id2label=self.id2label,
                label2id=self.label2id
            )
        except Exception as e:
            raise ValueError(f"Error in get Config Model method: {e}")
            
    def _get_model(self):
        try:
            return AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.model_config)
        except Exception as e:
            raise ValueError(f"Error in get Model method: {e}")

    def _get_tokenizer(self):
        try:
            return AutoTokenizer.from_pretrained(self.model_name)
        except Exception as e:
            raise ValueError(f"Error in get tokenizer method: {e}")
            
    def _get_metrics(self):
        metrics = self.config.get("metrics", {})
        self.metrics_to_compute = {metric: details["method"] for metric, details in metrics["type"].items() if details["compute"]}
        self.opt_metric = metrics.get("optimize", "")
        return self._get_compute_metric_f(metrics=self.metrics_to_compute)
    
    def _get_compute_metric_f(self, metrics: Dict[str, List[str]]):
        master_metric_mapper = {
            "f1_macro": f1_score,
            "f1_weighted": f1_score,
            "f1_binary": f1_score,
            "MulticlassAccuracy_micro": MulticlassAccuracy(average='micro', num_classes=4, k=1),
            "MulticlassAccuracy_macro": MulticlassAccuracy(average='macro', num_classes=4, k=1),
            "MulticlassAccuracy_None": MulticlassAccuracy(average=None, num_classes=4, k=1),
            "recall_micro": recall_score,
            "recall_macro": recall_score,
            "recall_weighted": recall_score,
            "recall_binary": recall_score,
            "precision_micro": precision_score,
            "precision_macro": precision_score,
            "precision_weighted": precision_score,
            "precision_binary": precision_score
        }
        self._metric_obj = {}
        for metric, methods in metrics.items():
            for method in methods:
                key_name = metric + "_" + method
                metric_obj = master_metric_mapper.get(key_name, None)
                if metric_obj is not None:
                    self._metric_obj[key_name] = metric_obj
        
        def compute_metrics(pred: EvalPrediction):
            predictions, labels = pred
            predictions = torch.tensor(predictions)
            labels = torch.tensor(labels)

            pred_label = torch.argmax(predictions, dim=1)
            true_label = torch.argmax(labels, dim=1)

            metric_results = {}
            for m_name, metric in self._metric_obj.items():
                metric_name, method = m_name.rsplit('_', 1)
                if metric_name in ["f1", "precision", "recall"]:
                    metric_results[m_name] = metric(y_true=true_label, y_pred=pred_label, average=method)
                elif metric_name == "MulticlassAccuracy":
                    metric.update(pred_label, true_label)
                    metric_results[m_name] = metric.compute()
            
            cross_entropy_loss_f = nn.CrossEntropyLoss()
            loss = cross_entropy_loss_f(predictions, labels)
            metric_results["Train Cross entropy loss"] = loss.item()
            metric_results["Inverse Train Cross entropy loss"] = 1 / loss.item()

            return metric_results
        return compute_metrics
    
    def _get_training_args(self):
        training_args = self.config.get("training_args", {})
        return TrainingArguments(
            training_args.get("dir_path", "./MODEL/DEFAULT"),
            evaluation_strategy=training_args.get("evaluation_strategy", "epoch"),
            report_to=training_args.get("report_to", "tensorboard"),
            save_strategy=training_args.get("save_strategy", "epoch"),
            learning_rate=training_args.get("learning_rate", 3e-05),
            per_device_train_batch_size=training_args.get("per_device_train_batch_size", 20),
            per_device_eval_batch_size=training_args.get("per_device_eval_batch_size", 32),
            num_train_epochs=training_args.get("num_train_epochs", 3),  # Ajustado para tiempos de prueba
            weight_decay=training_args.get("weight_decay", 0.01),
            adam_epsilon=training_args.get("adam_epsilon", 1e-08),
            load_best_model_at_end=training_args.get("load_best_model_at_end", True),
            metric_for_best_model=training_args.get("metric_for_best_model", self.opt_metric),
            gradient_accumulation_steps=training_args.get("gradient_accumulation_steps", 10),
            warmup_ratio=training_args.get("warmup_ratio", 0.03),
            fp16_full_eval=training_args.get("fp16_full_eval", True),
            fp16=training_args.get("fp16", True)
        )
        
    def _get_trainer(self, dataset: DatasetDict):
        trainer_config = self.config.get("trainer", {})
        callbacks_args = trainer_config.get("callbacks", {})
        early_stopping_args = callbacks_args.get("EarlyStoppingCallback", {})
        
        return Trainer(
            model=self.model,
            args=self.train_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=self.model_tokenizer,
            compute_metrics=self.compute_metric_f,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_args.get("early_stopping_patience", 3),
                    early_stopping_threshold=early_stopping_args.get("early_stopping_threshold", 0.0)
                ),
                TensorBoardCallback()
            ]
        )
    
    def train(self, dataset: DatasetDict):
        self.trainer = self._get_trainer(dataset)
        self.trainer.train()
    
    def predict(self, dataset: DatasetDict):
        try:
            return self.trainer.evaluate(dataset["test"])
        except Exception as e:
            raise ValueError(f"Error in predict method: {e}")

def create_synthetic_dataset(num_samples=1000):
    from datasets import Dataset as HFDataset  # Import necesario para evitar conflicto de nombre
    # Definimos algunas palabras comunes para generar los textos
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "and", "runs", "away"]
    labels = ["label1", "label2", "label3", "label4"]
    
    # Generamos textos aleatorios
    texts = [" ".join(np.random.choice(words, size=10)) for _ in range(num_samples)]
    
    # Generamos etiquetas aleatorias
    y = np.random.choice(labels, size=num_samples)
    
    # Creamos un DataFrame
    df = pd.DataFrame({"text": texts, "label": y})
    
    # Convertimos el DataFrame a un Dataset de HuggingFace
    dataset = HFDataset.from_pandas(df)
    
    # Dividimos el dataset en train, validation y test
    train_testvalid = dataset.train_test_split(test_size=0.3)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    
    # Creamos un DatasetDict
    dataset_dict = DatasetDict({
        'train': train_testvalid['train'],
        'validation': test_valid['train'],
        'test': test_valid['test']
    })
    
    return dataset_dict


import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig, EarlyStoppingCallback
)
from transformers.integrations import TensorBoardCallback
from transformers import EvalPrediction
from sklearn.metrics import f1_score, recall_score, precision_score
from torcheval.metrics import MulticlassAccuracy
from dotenv import load_dotenv
from typing import Dict, List, Tuple
from datasets import DatasetDict, load_dataset


# MODULE CLASS DOCU:
"""nn.Module :
Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes:
self.sub_module = nn.Linear(...)"""

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

# Util functions
def get_current_utc_date_iso():
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

class BoeNet:
    
    def __init__(self, model_conf_path: str):
        self.config_path = model_conf_path
        self.config = self._parse_config()
        self.model_name = self._get_model_name()
        self.model_tokenizer = self._get_tokenizer()
        self.model_config = self._get_model_config()
        self.model = self._get_model()
        self.compute_metric_f = self._get_metrics()
        self.train_args = self._get_training_args()

    def _parse_config(self) -> Dict:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        with open(self.config_path, 'r') as file:
            config = json.load(file)
        return config
    
    def _get_model_name(self):
        model = self.config.get("model", {})
        model_name = model.get("name", None)
        if model_name is None:
            raise ValueError("Model name not defined in the model config file")
        return model_name
    
    def _get_model_config(self):
        problem = self.config.get("problem", {})
        self.labels = problem.get("labels_to_pred", None)
        if self.labels is None:
            raise ValueError("Labels to predict not defined in the model config file")
        self.id2label = {i: lab for i, lab in enumerate(self.labels)}
        self.label2id = {lab: i for i, lab in enumerate(self.labels)}
        try:
            return AutoConfig.from_pretrained(
                pretrained_model_name_or_path=self.model_name,
                num_labels=len(self.labels),
                id2label=self.id2label,
                label2id=self.label2id
            )
        except Exception as e:
            raise ValueError(f"Error in get Config Model method: {e}")
            
    def _get_model(self):
        try:
            return AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.model_config)
        except Exception as e:
            raise ValueError(f"Error in get Model method: {e}")

    def _get_tokenizer(self):
        try:
            return AutoTokenizer.from_pretrained(self.model_name)
        except Exception as e:
            raise ValueError(f"Error in get tokenizer method: {e}")
            
    def _get_metrics(self):
        metrics = self.config.get("metrics", {})
        self.metrics_to_compute = {metric: details["method"] for metric, details in metrics["type"].items() if details["compute"]}
        self.opt_metric = metrics.get("optimize", "")
        return self._get_compute_metric_f(metrics=self.metrics_to_compute)
    
    def _get_compute_metric_f(self, metrics: Dict[str, List[str]]):
        master_metric_mapper = {
            "f1_macro": f1_score,
            "f1_weighted": f1_score,
            "f1_binary": f1_score,
            "MulticlassAccuracy_micro": MulticlassAccuracy(average='micro', num_classes=4, k=1),
            "MulticlassAccuracy_macro": MulticlassAccuracy(average='macro', num_classes=4, k=1),
            "MulticlassAccuracy_None": MulticlassAccuracy(average=None, num_classes=4, k=1),
            "recall_micro": recall_score,
            "recall_macro": recall_score,
            "recall_weighted": recall_score,
            "recall_binary": recall_score,
            "precision_micro": precision_score,
            "precision_macro": precision_score,
            "precision_weighted": precision_score,
            "precision_binary": precision_score
        }
        self._metric_obj = {}
        for metric, methods in metrics.items():
            for method in methods:
                key_name = metric + "_" + method
                metric_obj = master_metric_mapper.get(key_name, None)
                if metric_obj is not None:
                    self._metric_obj[key_name] = metric_obj
        
        def compute_metrics(pred: EvalPrediction):
            predictions, labels = pred
            predictions = torch.tensor(predictions)
            labels = torch.tensor(labels)

            pred_label = torch.argmax(predictions, dim=1)
            true_label = torch.argmax(labels, dim=1)

            metric_results = {}
            for m_name, metric in self._metric_obj.items():
                metric_name, method = m_name.rsplit('_', 1)
                if metric_name in ["f1", "precision", "recall"]:
                    metric_results[m_name] = metric(y_true=true_label, y_pred=pred_label, average=method)
                elif metric_name == "MulticlassAccuracy":
                    metric.update(pred_label, true_label)
                    metric_results[m_name] = metric.compute()
            
            cross_entropy_loss_f = nn.CrossEntropyLoss()
            loss = cross_entropy_loss_f(predictions, labels)
            metric_results["Train Cross entropy loss"] = loss.item()
            metric_results["Inverse Train Cross entropy loss"] = 1 / loss.item()

            return metric_results
        return compute_metrics
    
    def _get_training_args(self):
        training_args = self.config.get("training_args", {})
        return TrainingArguments(
            training_args.get("dir_path", "./MODEL/DEFAULT"),
            evaluation_strategy=training_args.get("evaluation_strategy", "epoch"),
            report_to=training_args.get("report_to", "tensorboard"),
            save_strategy=training_args.get("save_strategy", "epoch"),
            learning_rate=training_args.get("learning_rate", 3e-05),
            per_device_train_batch_size=training_args.get("per_device_train_batch_size", 20),
            per_device_eval_batch_size=training_args.get("per_device_eval_batch_size", 32),
            num_train_epochs=training_args.get("num_train_epochs", 3),  # Ajustado para tiempos de prueba
            weight_decay=training_args.get("weight_decay", 0.01),
            adam_epsilon=training_args.get("adam_epsilon", 1e-08),
            load_best_model_at_end=training_args.get("load_best_model_at_end", True),
            metric_for_best_model=training_args.get("metric_for_best_model", self.opt_metric),
            gradient_accumulation_steps=training_args.get("gradient_accumulation_steps", 10),
            warmup_ratio=training_args.get("warmup_ratio", 0.03),
            fp16_full_eval=training_args.get("fp16_full_eval", True),
            fp16=training_args.get("fp16", True)
        )
        
    def _get_trainer(self, dataset: DatasetDict):
        trainer_config = self.config.get("trainer", {})
        callbacks_args = trainer_config.get("callbacks", {})
        early_stopping_args = callbacks_args.get("EarlyStoppingCallback", {})
        
        return Trainer(
            model=self.model,
            args=self.train_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=self.model_tokenizer,
            compute_metrics=self.compute_metric_f,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_args.get("early_stopping_patience", 3),
                    early_stopping_threshold=early_stopping_args.get("early_stopping_threshold", 0.0)
                ),
                TensorBoardCallback()
            ]
        )
    
    def train(self, dataset: DatasetDict):
        self.trainer = self._get_trainer(dataset)
        self.trainer.train()
    
    def predict(self, dataset: DatasetDict):
        try:
            return self.trainer.evaluate(dataset["test"])
        except Exception as e:
            raise ValueError(f"Error in predict method: {e}")

def create_synthetic_dataset(tokenizer , num_samples=1000):
    tokenizer = tokenizer
    from datasets import Dataset as HFDataset  # Import necesario para evitar conflicto de nombre
    # Definimos algunas palabras comunes para generar los textos
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "and", "runs", "away"]
    labels = ["label1", "label2", "label3", "label4"]
    
    # Generamos textos aleatorios
    texts = [" ".join(np.random.choice(words, size=10)) for _ in range(num_samples)]
    
    # Generamos etiquetas aleatorias
    y = np.random.choice(labels, size=num_samples)
    
    # Creamos un DataFrame
    df = pd.DataFrame({"text": texts, "label": y})
    
    # Convertimos el DataFrame a un Dataset de HuggingFace
    dataset = HFDataset.from_pandas(df)
    
    # Dividimos el dataset en train, validation y test
    train_testvalid = dataset.train_test_split(test_size=0.3)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    
    # Creamos un DatasetDict
    dataset_dict = DatasetDict({
        'train': train_testvalid['train'],
        'validation': test_valid['train'],
        'test': test_valid['test']
    })
    
    def _process_dataset(dataset) -> dict:
        label_f_name = 'label'
        text_f_name = 'text'
        text = str(dataset[text_f_name]) # aseguramos tipo de dato es str
        
        # tokenizacion
        tokenized = tokenizer(text, padding=False, truncation=True)

        # labels
        tokenized["labels"] = dataset[label_f_name]

        return tokenized

    dataset_tokenize = dataset.map(_process_dataset, batched=False, remove_columns=df.columns.tolist())
    return dataset_tokenize


if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    config_path = 'C:\\Users\\Jorge\\Desktop\\MASTER_IA\\TFM\\proyectoCHROMADB\\config\\model.json' 
    boenet = BoeNet(config_path)
    
    synthetic_dataset = create_synthetic_dataset(tokenizer =boenet.model_tokenizer )
    print(synthetic_dataset)
    boenet.train(synthetic_dataset)
    results = boenet.predict(synthetic_dataset)
    print(results)





