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
from ts.torch_handler.captum_utils import construct_classification_input_ref_pair,construct_attention_mask,summarize_attributions,squad_pos_forward_func,construct_input_ref_pair,text_visualization, squad_cls_forward_func

# from captum_utils import construct_classification_input_ref_pair,construct_attention_mask,summarize_attributions,squad_pos_forward_func,construct_input_ref_pair,text_visualization, squad_cls_forward_func

from enum import Enum

class WordImportance():

    def __init__(self, model):
        self.model = model


    def get_insight(self,tokenizer,setup_config,request,mapping):
        self.tokenizer = tokenizer
        self.config = setup_config
        self.mapping = mapping
        self.request = request

        self.ref_token_id = self.tokenizer.pad_token_id # A token used for generating token reference
        self.sep_token_id = self.tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
        self.cls_token_id = self.tokenizer.cls_token_id

        interperation = self.request["interperation"]
        model_embedding_class = self.request["model_embedding_class"]
        embed = getattr(self.model, model_embedding_class)
        embedding = embed.embeddings
        #preprocessing text for sequence_classification and token_classification.
        if self.config["mode"]== "sequence_classification" or self.config["mode"]== "token_classification" :
            # if self.captum_visualization:
            query_text = self.request["text"]
            input_ids, ref_input_ids = construct_classification_input_ref_pair(self.tokenizer, query_text, self.ref_token_id, self.sep_token_id, self.cls_token_id)
            indices = input_ids[0].detach().tolist()
            all_tokens = self.tokenizer.convert_ids_to_tokens(indices)
            attention_mask = construct_attention_mask(input_ids)
            preds = self.model(input_ids,attention_mask)

            if self.config["mode"]== "sequence_classification":

                target = int(self.request["target_class"])
                pred_probability = torch.max(torch.softmax(preds[0], dim=0))
                pred_class = self.mapping[str(preds[0].argmax(1).item())]
                true_class = -1
                attr_class = self.mapping[str(target)]
                print("#################################",type(pred_probability), type(pred_class))
            elif self.config["mode"]== "token_classification":

                label_list = self.mapping["label_list"]
                label_list = label_list.strip('][').split(', ')
                target =(int(self.request["target_token"]),int(self.request["target_class"]))
                pred_probability = torch.max(torch.softmax(preds[0][target[0]][target[1]], dim=0))
                pred_class = label_list[torch.argmax(preds[0][target[0]][target[1]])]
                true_class = -1
                attr_class = label_list[target[1]]


            lig = LayerIntegratedGradients(squad_cls_forward_func, embedding)
            attributions_ig, delta = lig.attribute(inputs=input_ids, baselines=ref_input_ids,
                                       additional_forward_args=(self.model,attention_mask),
                                       target=target,
                                       n_steps=20,
                                       return_convergence_delta=True)
            attributions_sequence_cls_sum = summarize_attributions(attributions_ig)
            attr_score = attributions_sequence_cls_sum.sum()
            if self.request["output"] == "html":
                classification_vis = viz.VisualizationDataRecord(
                            attributions_sequence_cls_sum,
                            pred_probability,
                            pred_class,
                            true_class,
                            attr_class,
                            attributions_sequence_cls_sum.sum(),
                            all_tokens,
                            delta)
                print("**************delta**********",delta)
                dom = text_visualization([classification_vis])
                return dom
            elif self.request["output"] == "json":
                return attributions_sequence_cls_sum.tolist()
        #preprocessing text for question_answering.
        elif self.config["mode"]== "question_answering":
            #TODO Reading the context from a pickeled file or other fromats that
            # fits the requirements of the task in hand. If this is done then need to
            # modify the following preprocessing accordingly.

            # the sample text for question_answering in the current version
            # should be formated as dictionary with question and text as keys
            # and related text as values.
            # we use this format here seperate question and text for encoding.

            question = self.request["question"]
            context = self.request["context"]
            all_input =question+context
            # if self.captum_visualization:
            input_ids, ref_input_ids, sep_id = construct_input_ref_pair(self.tokenizer,question, context, self.ref_token_id, self.sep_token_id, self.cls_token_id)
            indices = input_ids[0].detach().tolist()
            all_tokens = self.tokenizer.convert_ids_to_tokens(indices)
            attention_mask = construct_attention_mask(input_ids)
            lig = LayerIntegratedGradients(squad_pos_forward_func, embedding)

            attributions_start, delta_start = lig.attribute(inputs=input_ids,
                                              baselines=ref_input_ids,
                                              additional_forward_args=(self.model,attention_mask, 0),
                                              return_convergence_delta=True)
            attributions_end, delta_end = lig.attribute(inputs=input_ids, baselines=ref_input_ids,
                                            additional_forward_args=(self.model,attention_mask, 1),
                                            return_convergence_delta=True)
            attributions_start_sum = summarize_attributions(attributions_start)
            attributions_end_sum = summarize_attributions(attributions_end)

            start_scores, end_scores = self.model(input_ids,attention_mask)
            if self.request["output"] == "html":

                start_position_vis = viz.VisualizationDataRecord(
                            attributions_start_sum,
                            torch.max(torch.softmax(start_scores[0], dim=0)),
                            torch.argmax(start_scores),
                            -1,
                            torch.argmax(start_scores),
                            attributions_start_sum.sum(),
                            all_tokens,
                            delta_start)
                dom = text_visualization([start_position_vis])
                end_position_vis = viz.VisualizationDataRecord(
                            attributions_end_sum,
                            torch.max(torch.softmax(end_scores[0], dim=0)),
                            torch.argmax(end_scores),
                            -1,
                            torch.argmax(end_scores),
                            attributions_end_sum.sum(),
                            all_tokens,
                            delta_start)
                start_dom = text_visualization([start_position_vis])
                end_dom = text_visualization([end_position_vis])
                dom = start_dom + end_dom
                return dom
            elif self.request["output"] == "json":
                return [attributions_start_sum.tolist(), attributions_end_sum.tolist()]
