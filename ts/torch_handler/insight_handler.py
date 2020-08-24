import os
import ast
import torch
import json
import logging
import sys
import tempfile
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Iterable, List, Tuple, Union
from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
from ts.torch_handler import captum_utils
# from ts.utils import captum_utils
from enum import Enum
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForQuestionAnswering,AutoModelForTokenClassification,AutoConfig

from ts.torch_handler.word_importance import WordImportance
# class captum_insight():
# setup_config = {
#  "model_name":"bert-base-uncased",
#  "mode":"sequence_classification",
#  "do_lower_case":"True",
#  "num_labels":"2",
#  "save_mode":"torchscript",
#  "max_length":"150"
# }
# mapping = {
#  "0":"Not Accepted",
#  "1":"Accepted"
# }
# mode = setup_config["mode"]
# pretrained_model_name = setup_config["model_name"]
# num_labels = int(setup_config["num_labels"])
# save_mode = setup_config["save_mode"]
# max_length = setup_config["max_length"]
# do_lower_case = setup_config["do_lower_case"]
# if save_mode == "torchscript":
#     torchscript = True
# else:
#     torchscript = False
#
# if mode== "sequence_classification":
#     config = AutoConfig.from_pretrained(pretrained_model_name,num_labels=num_labels,torchscript=torchscript)
#     model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, config=config)
#     tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name,do_lower_case=do_lower_case)
# elif mode== "question_answering":
#     config = AutoConfig.from_pretrained(pretrained_model_name,torchscript=torchscript)
#     model = AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name,config=config)
#     tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name,do_lower_case=do_lower_case)
# elif mode== "token_classification":
#     config= AutoConfig.from_pretrained(pretrained_model_name,num_labels=num_labels,torchscript=torchscript)
#     model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name, config=config)
#     tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name,do_lower_case=do_lower_case)
#
# request = {"text":"Bloomberg has decided to publish a new report on the global economy, this seems to be a good signal at the end of the day that the world is moving to a brighter side and every other company is trying to increase their stability which is great signal to the whole world economy","interperation":"True", "output":"html", "target_class":"1", "model_embedding_class":"bert"}

class InsightMethod(Enum):
    word_importance = "WordImportance"


    def __str__(self):
        return self.value

class Insight():

    def __init__(self, method:InsightMethod,service, request):
        self.insight_method = method
        print("method: ",self.insight_method)
        self.service = service
        self.request = request
        # self.tokenizer = service.tokenizer
        # self.setup_config = service.setup_config
        # self.mapping = service.mapping
        # self.request = request
        if self.insight_method is InsightMethod.word_importance:
            print("comparing methods: ",self.insight_method, InsightMethod.word_importance)

            self.attribution = WordImportance(self.service.model)


    def insight(self,**args):
        try:
            if self.insight_method is InsightMethod.word_importance:
                dom = self.attribution.get_insight(tokenizer = self.service.tokenizer, setup_config = self.service.setup_config,request = self.request, mapping = self.service.mapping)
                return dom
        except Exception as e:
            raise e

#
# if __name__ == "__main__":
#     if request["interperation"]:
#         insight_handler = Insight(InsightMethod("WordImportance"),model)
#         insight_handler.insight(tokenizer = tokenizer, setup_config = setup_config,request = request, mapping = mapping)
