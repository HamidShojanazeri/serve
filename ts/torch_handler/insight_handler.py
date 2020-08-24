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
