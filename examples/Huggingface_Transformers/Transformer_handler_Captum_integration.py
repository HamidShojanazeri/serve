from abc import ABC
import json
import logging
import os
import ast
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForQuestionAnswering,AutoModelForTokenClassification
from ts.torch_handler.base_handler import BaseHandler
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
logger = logging.getLogger(__name__)

class TransformersSeqClassifierHandler(BaseHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """
    def __init__(self):
        super(TransformersSeqClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        #read configs for the mode, model_name, etc. from setup_config.json
        setup_config_path = os.path.join(model_dir, "setup_config.json")
        captum_utils = os.path.join(model_dir, "captum_utils.py")
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                self.setup_config = json.load(setup_config_file)
        else:
            logger.warning('Missing the setup_config.json file.')

        #Loading the model and tokenizer from checkpoint and config files based on the user's choice of mode
        #further setup config can be added.
        if self.setup_config["save_mode"] == "torchscript":
            self.model = torch.jit.load(model_pt_path)
        elif self.setup_config["save_mode"] == "pretrained":
            if self.setup_config["mode"]== "sequence_classification":
                self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            elif self.setup_config["mode"]== "question_answering":
                self.model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
            elif self.setup_config["mode"]== "token_classification":
                self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
            else:
                logger.warning('Missing the operation mode.')
        else:
            logger.warning('Missing the checkpoint or state_dict.')

        if not os.path.isfile(os.path.join(model_dir, "vocab.*")):
            self.tokenizer = AutoTokenizer.from_pretrained(self.setup_config["model_name"],do_lower_case=self.setup_config["do_lower_case"])
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir,do_lower_case=self.setup_config["do_lower_case"])

        self.model.to(self.device)
        self.model.eval()

        logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))
        #intilaizing rquired token references for captum
        self.captum_visualization = self.setup_config["captum_visualization"]
        self.captum_target_class = self.setup_config["captum_target_class"]
        if self.captum_visualization:
            self.ref_token_id = self.tokenizer.pad_token_id # A token used for generating token reference

            self.sep_token_id = self.tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.

            self.cls_token_id = self.tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence
        # Read the mapping file, index to object name
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        # Question answering does not need the index_to_name.json file.
        if not self.setup_config["mode"]== "question_answering":
            if os.path.isfile(mapping_file_path):
                with open(mapping_file_path) as f:
                    self.mapping = json.load(f)
            else:
                logger.warning('Missing the index_to_name.json file.')

        self.initialized = True


    def preprocess(self, data):
        """ Basic text preprocessing, based on the user's chocie of application mode.
        """

        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        input_text = text.decode('utf-8')
        Body = ast.literal_eval(input_text)
        max_length = self.setup_config["max_length"]
        logger.info("Received text: '%s'", input_text)
        #preprocessing text for sequence_classification and token_classification.
        if self.setup_config["mode"]== "sequence_classification" or self.setup_config["mode"]== "token_classification" :
            query_text = Body["text"]
            inputs = self.tokenizer.encode_plus(query_text,max_length = int(max_length),pad_to_max_length = True, add_special_tokens = True, return_tensors = 'pt')

        elif self.setup_config["mode"]== "question_answering":
            #TODO Reading the context from a pickeled file or other fromats that
            # fits the requirements of the task in hand. If this is done then need to
            # modify the following preprocessing accordingly.

            # the sample text for question_answering in the current version
            # should be formated as dictionary with question and text as keys
            # and related text as values.
            # we use this format here seperate question and text for encoding.
            question = Body["question"]
            context = Body["context"]
            inputs = self.tokenizer.encode_plus(question, context,max_length = int(max_length),pad_to_max_length = True, add_special_tokens=True, return_tensors="pt")
        return inputs, Body

    def interpret(self, Body):
        from captum_utils import construct_classification_input_ref_pair,construct_attention_mask,summarize_attributions,squad_pos_forward_func,construct_input_ref_pair,text_visualization

        interperation = Body["interperation"]
        # max_length = self.setup_config["max_length"]
        model_embedding_class = self.setup_config["model_embedding_class"]
        embed = getattr(self.model, model_embedding_class)
        embedding = embed.embeddings
        if self.captum_target_class:
            target = int(self.captum_target_class)
        else:
            target = None
        # logger.info("Received text: '%s'", input_text)
        #preprocessing text for sequence_classification and token_classification.
        if self.setup_config["mode"]== "sequence_classification" or self.setup_config["mode"]== "token_classification" :
            # if self.captum_visualization:
            query_text = Body["text"]
            input_ids, ref_input_ids = construct_classification_input_ref_pair(self.tokenizer, query_text, self.ref_token_id, self.sep_token_id, self.cls_token_id)
            indices = input_ids[0].detach().tolist()
            all_tokens = self.tokenizer.convert_ids_to_tokens(indices)
            print("##############",len(all_tokens),input_ids.size())
            attention_mask = construct_attention_mask(input_ids)
            lig = LayerIntegratedGradients(squad_pos_forward_func, embedding)
            target =(0,1)
            attributions_ig, delta = lig.attribute(inputs=input_ids, baselines=ref_input_ids,
                                       additional_forward_args=(self.model,attention_mask),
                                       target=target,
                                       return_convergence_delta=True)

            attributions_sequence_cls_sum = summarize_attributions(attributions_ig)
            print("***********", attributions_sequence_cls_sum)
            preds = self.model(input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(preds[0], dim=2)
            print("@@@@@@@@@@@@@@@@@@@@@@@@",preds[0][0].size(),attributions_ig.size())
            print("####################### all tokens and attribute scores",len(all_tokens),attributions_ig.size(), input_ids.size())
            classification_vis = viz.VisualizationDataRecord(
                        attributions_sequence_cls_sum,
                        torch.max(torch.softmax(preds[0][0][1], dim=0)),
                        torch.argmax(preds[0][0][1]),
                        torch.argmax(preds[0][0][1]),
                        str(-1),
                        attributions_sequence_cls_sum.sum(),
                        all_tokens,
                        delta)
            dom = text_visualization([classification_vis])
            print(dom)

        #preprocessing text for question_answering.
        elif self.setup_config["mode"]== "question_answering":
            #TODO Reading the context from a pickeled file or other fromats that
            # fits the requirements of the task in hand. If this is done then need to
            # modify the following preprocessing accordingly.

            # the sample text for question_answering in the current version
            # should be formated as dictionary with question and text as keys
            # and related text as values.
            # we use this format here seperate question and text for encoding.

            question = Body["question"]
            context = Body["context"]
            all_input =question+context
            # if self.captum_visualization:
            input_ids, ref_input_ids, sep_id = construct_input_ref_pair(self.tokenizer,question, context, self.ref_token_id, self.sep_token_id, self.cls_token_id)
            # ref_input_new = self.construct_qa_input_ref_pair(question,context, self.ref_token_id, self.sep_token_id, self.cls_token_id)
            # input_ids = inputs['input_ids']
            # attention_mask = inputs['attention_mask']
            # ref_ids = self.pad_sequence(ref_input_new,max_length)
            indices = input_ids[0].detach().tolist()
            all_tokens = self.tokenizer.convert_ids_to_tokens(indices)
            attention_mask = construct_attention_mask(input_ids)
            lig = LayerIntegratedGradients(squad_pos_forward_func, embedding)
            print("############## Input IDSSSSSSSS",input_ids)
            print("############## REFFFFFFF IDSSSSSSSs",ref_input_ids)

            attributions_start, delta_start = lig.attribute(inputs=input_ids,
                                              baselines=ref_input_ids,
                                              additional_forward_args=(self.model,attention_mask, 0),
                                              return_convergence_delta=True)
            attributions_end, delta_end = lig.attribute(inputs=input_ids, baselines=ref_input_ids,
                                            additional_forward_args=(self.model,attention_mask, 1),
                                            return_convergence_delta=True)
            attributions_start_sum = summarize_attributions(attributions_start)
            attributions_end_sum = summarize_attributions(attributions_end)
            print("***********", attributions_start_sum)

            start_scores, end_scores = self.model(input_ids, attention_mask=attention_mask)
            print("@@@@@@@@@@@@@@@@@@@@@@@@",start_scores.size(),end_scores.size(), "torchmax : ",torch.max(torch.softmax(start_scores[0], dim=0)), "Start score[0]", start_scores[0].size())
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
            print(type(dom), dom)
            return start_dom, end_dom
        return

    def inference(self, inputs, Body):
        """ Predict the class (or classes) of the received text using the serialized transformers checkpoint.
        """
        response = {}
        if Body["interperation"]:
            start_dom, end_dom = self.interpret(Body)
            dom = start_dom + end_dom
            tmpdir = tempfile.mkdtemp()
            filename = 'visualization.html'

            # Ensure the file is read/write by the creator only
            # saved_umask = os.umask(0o77)
            if Body["output"]=="html":
                path = os.path.join(tmpdir,filename)
                print(path)
                try:
                    with open(path, "w") as tmp:
                        for line in dom:
                            tmp.write(line)
                    response["visuliazation"] = path
                except IOError as e:
                    print('IOError')
            elif Body["output"]=="json":
                response["visuliazation"] = dom


        input_ids = inputs["input_ids"].to(self.device)
        # Handling inference for sequence_classification.
        if self.setup_config["mode"]== "sequence_classification":
            predictions = self.model(input_ids)
            prediction = predictions[0].argmax(1).item()

            logger.info("Model predicted: '%s'", prediction)

            if self.mapping:
                prediction = self.mapping[str(prediction)]
        # Handling inference for question_answering.
        elif self.setup_config["mode"]== "question_answering":
            # the output should be only answer_start and answer_end
            # we are outputing the words just for demonstration.
            answer_start_scores, answer_end_scores = self.model(input_ids)
            answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
            input_ids = inputs["input_ids"].tolist()[0]
            prediction = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

            logger.info("Model predicted: '%s'", prediction)
        # Handling inference for token_classification.
        elif self.setup_config["mode"]== "token_classification":
            outputs = self.model(input_ids)[0]
            predictions = torch.argmax(outputs, dim=2)
            tokens = self.tokenizer.tokenize(self.tokenizer.decode(inputs["input_ids"][0]))
            if self.mapping:
                label_list = self.mapping["label_list"]
            label_list = label_list.strip('][').split(', ')
            prediction = [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].tolist())]

            logger.info("Model predicted: '%s'", prediction)
        response["prediction"]= prediction
        return [response]

    def postprocess(self, inference_output):
        # TODO: Add any needed post-processing of the model predictions here
        return inference_output


_service = TransformersSeqClassifierHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data,Body = _service.preprocess(data)
        data = _service.inference(data, Body)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e
