import os
import logging
from typing import Union, Optional, Callable, ClassVar, TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from datasets import Dataset 

# Logging configuration
logger = logging.getLogger(__name__)

class RagasDataset(BaseModel):
    """
    Example :
    data_samples = {
    'question': ['When was the first super bowl?', 'Who won the most super bowls?'],
    'answer': ['The first superbowl was held on January 15, 1967', 'The most super bowls have been won by The New England Patriots'],
    'contexts' : [['The Super Bowl....season since 1966,','replacing the NFL...in February.'], 
    ['The Green Bay Packers...Green Bay, Wisconsin.','The Packers compete...Football Conference']],
    'ground_truth': ['The first superbowl was held on January 15, 1967', 'The New England Patriots have won the Super Bowl a record six times']
    }
    """
    
    question : list[str]
    answer : list[str]
    contexts : list[list[str]]
    ground_truth : list[str]
    
    def add_atributes(self, question : str, answer : str, contexts : list[str], ground_truth : str):
        self.question.append(question)
        self.answer.append(question)
        self.contexts.append(question)
        self.ground_truth.append(question)
        
    def to_dataset(self) -> Dataset:
        self.dataset = Dataset.from_dict(self.model_dump(mode="python")) 
        return self.dataset
    