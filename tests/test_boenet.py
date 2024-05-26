import unittest
from unittest.mock import patch, mock_open
import json
import os
import torch
from datasets import DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from src.package.module.BOENet import BoeNet, create_synthetic_dataset 

#Datos de configuración simulados
fake_config_data = json.dumps({
    "model": {"name": "bert-base-uncased"},
    "problem": {"labels_to_pred": ["label1", "label2", "label3", "label4"]},
    "metrics": {
        "type": {"f1": {"method": ["macro"], "compute": True}},
        "optimize": "f1_macro"
    },
    "training_args": {
        "dir_path": "./MODEL/DEFAULT",
        "evaluation_strategy": "epoch",
        "report_to": "tensorboard",
        "save_strategy": "epoch",
        "learning_rate": 3e-05,
        "per_device_train_batch_size": 20,
        "per_device_eval_batch_size": 32,
        "num_train_epochs": 3,
        "weight_decay": 0.01,
        "adam_epsilon": 1e-08,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "gradient_accumulation_steps": 10,
        "warmup_ratio": 0.03,
        "fp16_full_eval": False,
        "fp16": False
    }
})

class TestBoeNet(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open, read_data=fake_config_data)
    @patch("os.path.exists", return_value=True)
    def test_parse_config(self, mock_exists, mock_file):
        boenet = BoeNet("fake_config_path")
        self.assertEqual(boenet.config["model"]["name"], "bert-base-uncased")
        self.assertEqual(boenet.config["problem"]["labels_to_pred"], ["label1", "label2", "label3", "label4"])

    @patch("builtins.open", new_callable=mock_open, read_data=fake_config_data)
    @patch("os.path.exists", return_value=True)
    @patch("transformers.AutoModelForSequenceClassification.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained", return_value=AutoTokenizer.from_pretrained("bert-base-uncased"))
    def test_model_initialization(self, mock_tokenizer, mock_model, mock_exists, mock_file):
        mock_model.return_value = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        
        boenet = BoeNet("fake_config_path")
        self.assertIsNotNone(boenet.model)
        self.assertIsNotNone(boenet.model_tokenizer)

    @patch("builtins.open", new_callable=mock_open, read_data=fake_config_data)
    @patch("os.path.exists", return_value=True)
    @patch("transformers.AutoTokenizer.from_pretrained", return_value=AutoTokenizer.from_pretrained("bert-base-uncased"))
    def test_training_args(self, mock_tokenizer, mock_exists, mock_file):
        boenet = BoeNet("fake_config_path")
        train_args = boenet._get_training_args()
        self.assertIsInstance(train_args, TrainingArguments)
        self.assertEqual(train_args.num_train_epochs, 3)

    def test_create_synthetic_dataset(self):
        dataset = create_synthetic_dataset()
        self.assertIsInstance(dataset, DatasetDict)
        self.assertIn("train", dataset)
        self.assertIn("validation", dataset)
        self.assertIn("test", dataset)

    @patch("builtins.open", new_callable=mock_open, read_data=fake_config_data)
    @patch("os.path.exists", return_value=True)
    @patch("transformers.Trainer.train")
    @patch("transformers.Trainer.evaluate")
    @patch("transformers.AutoTokenizer.from_pretrained", return_value=AutoTokenizer.from_pretrained("bert-base-uncased"))
    def test_train_and_predict(self, mock_tokenizer, mock_evaluate, mock_train, mock_exists, mock_file):
        mock_train.return_value = None
        mock_evaluate.return_value = {"f1_macro": 0.8}
        
        boenet = BoeNet("fake_config_path")
        dataset = create_synthetic_dataset()
        boenet.train(dataset)
        results = boenet.predict(dataset)
        
        self.assertIn("f1_macro", results)
        self.assertEqual(results["f1_macro"], 0.8)

if __name__ == "__main__":
    unittest.main()