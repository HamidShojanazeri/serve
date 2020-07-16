from abc import ABC
import json
import logging
import os
import ast
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForQuestionAnswering,AutoModelForTokenClassification
from ts.torch_handler.base_handler import BaseHandler
import sys
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
device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")


try:
    from IPython.core.display import display, HTML

    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

def construct_input_ref_pair(tokenizer,question, text, ref_token_id, sep_token_id, cls_token_id):
      question_ids = tokenizer.encode(question, add_special_tokens=False)
      text_ids = tokenizer.encode(text, add_special_tokens=False)
      # construct input token ids
      input_ids = [cls_token_id] + question_ids + [sep_token_id] + text_ids + [sep_token_id]
      print("question ids",input_ids)

      # construct reference token ids
      ref_input_ids = [cls_token_id] + [ref_token_id] * len(question_ids) + [sep_token_id] + \
          [ref_token_id] * len(text_ids) + [sep_token_id]

      return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(question_ids)

def construct_qa_input_ref_pair(tokenizer,question, text, ref_token_id, sep_token_id, cls_token_id):
  question_ids = tokenizer.encode(question, add_special_tokens=False)
  text_ids = tokenizer.encode(text, add_special_tokens=False)
  # construct reference token ids
  ref_input_ids = [cls_token_id] + [ref_token_id] * len(question_ids) + [sep_token_id] + \
      [ref_token_id] * len(text_ids) + [sep_token_id]

  return torch.tensor([ref_input_ids], device= device)

def construct_classification_input_ref_pair(tokenizer, text, ref_token_id, sep_token_id, cls_token_id):
  text_ids = tokenizer.encode(text, add_special_tokens=False)
  input_ids = [cls_token_id] + text_ids + [sep_token_id]
  # construct reference token ids
  ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]

  return torch.tensor([input_ids], device= device), torch.tensor([ref_input_ids], device= device)

def construct_attention_mask(input_ids):
  return torch.ones_like(input_ids)


def squad_cls_forward_func(inputs,model=None, attention_mask=None):
    model = model
    pred = model(inputs, attention_mask=attention_mask)
    return pred[0]

def squad_pos_forward_func(inputs,model=None, attention_mask=None, position=0):
    model = model
    pred = model(inputs, attention_mask=attention_mask)
    pred = pred[position]
    return pred.max(1).values

def summarize_attributions(attributions):
  attributions = attributions.sum(dim=-1).squeeze(0)
  attributions = attributions / torch.norm(attributions)
  return attributions

def pad_sequence(unpadded_seq,max_length):
  unpadded_seq_list = unpadded_seq.squeeze(0).tolist()
  max_range = int(max_length) - len(unpadded_seq_list)
  for i in range(max_range):
      unpadded_seq_list.append(0)
  return torch.tensor([unpadded_seq_list])


def text_visualization(datarecords: Iterable[viz.VisualizationDataRecord]) -> None:
    assert HAS_IPYTHON, (
        "IPython must be available to visualize text. "
        "Please run 'pip install ipython'."
    )
    dom = ["<table width: 100%>"]
    rows = [
        "<th>Predicted Label</th>"
        "<th>Attribution Label</th>"
        "<th>Attribution Score</th>"
        "<th>Word Importance</th>"
    ]
    for datarecord in datarecords:
        rows.append(
            "".join(
                [
                    "<tr>",
                    # viz.format_classname(datarecord.true_class),
                    viz.format_classname(
                        "{0} ({1:.2f})".format(
                            datarecord.pred_class, datarecord.pred_prob
                        )
                    ),
                    viz.format_classname(datarecord.attr_class),
                    viz.format_classname("{0:.2f}".format(datarecord.attr_score)),
                    viz.format_word_importances(
                        datarecord.raw_input, datarecord.word_attributions
                    ),
                    "<tr>",
                ]
            )
        )

    dom.append("".join(rows))
    dom.append("</table>")
    display(HTML("".join(dom)))
    return dom
