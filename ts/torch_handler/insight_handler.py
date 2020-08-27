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
import importlib
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Iterable, List, Tuple, Union
from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
from ts.torch_handler import captum_utils
# from ts.utils import captum_utils
from enum import Enum
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForQuestionAnswering,AutoModelForTokenClassification,AutoConfig
import importlib.machinery


class Insight():

    def __init__(self,service):
        self.service = service
        directory, module_name = os.path.split(self.service.captum_handler_path)
        module_name = os.path.splitext(module_name)[0]
        self.captum_handler = __import__(module_name)
