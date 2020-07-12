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
        captum_utilss = os.path.join(model_dir, "captum_utilss.py")
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
    def predict(self,inputs,attention_mask=None):
        return self.model(inputs, attention_mask=attention_mask, )

    def squad_pos_forward_func(self,inputs, attention_mask=None, position=0):
        pred = self.predict(inputs,
                       attention_mask=attention_mask)
        pred = pred[position]
        return pred.max(1).values

    # def construct_input_ref_pair(self,question, text, ref_token_id, sep_token_id, cls_token_id):
    #     question_ids = self.tokenizer.encode(question, add_special_tokens=False)
    #     text_ids = self.tokenizer.encode(text, add_special_tokens=False)
    #     # construct input token ids
    #     input_ids = [cls_token_id] + question_ids + [sep_token_id] + text_ids + [sep_token_id]
    #     print("question ids",input_ids)
    #
    #     # construct reference token ids
    #     ref_input_ids = [cls_token_id] + [ref_token_id] * len(question_ids) + [sep_token_id] + \
    #         [ref_token_id] * len(text_ids) + [sep_token_id]
    #
    #     return torch.tensor([input_ids], device=self.device), torch.tensor([ref_input_ids], device=self.device), len(question_ids)
    #
    # def construct_qa_input_ref_pair(self,question, text, ref_token_id, sep_token_id, cls_token_id):
    #     question_ids = self.tokenizer.encode(question, add_special_tokens=False)
    #     text_ids = self.tokenizer.encode(text, add_special_tokens=False)
    #     # construct reference token ids
    #     ref_input_ids = [cls_token_id] + [ref_token_id] * len(question_ids) + [sep_token_id] + \
    #         [ref_token_id] * len(text_ids) + [sep_token_id]
    #
    #     return torch.tensor([ref_input_ids], device=self.device)
    #
    # def construct_classification_input_ref_pair(self, text, ref_token_id, sep_token_id, cls_token_id):
    #     text_ids = self.tokenizer.encode(text, add_special_tokens=False)
    #     input_ids = [cls_token_id] + text_ids + [sep_token_id]
    #     # construct reference token ids
    #     ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]
    #
    #     return torch.tensor([input_ids], device=self.device), torch.tensor([ref_input_ids], device=self.device)
    #
    # def construct_attention_mask(self,input_ids):
    #     return torch.ones_like(input_ids)
    #
    # def summarize_attributions(self,attributions):
    #     attributions = attributions.sum(dim=-1).squeeze(0)
    #     attributions = attributions / torch.norm(attributions)
    #     return attributions
    #
    # def pad_sequence(self,unpadded_seq,max_length):
    #     unpadded_seq_list = unpadded_seq.squeeze(0).tolist()
    #     max_range = int(max_length) - len(unpadded_seq_list)
    #     for i in range(max_range):
    #         unpadded_seq_list.append(0)
    #     return torch.tensor([unpadded_seq_list])

    def preprocess(self, data):
        """ Basic text preprocessing, based on the user's chocie of application mode.
        """
        from captum_utilss import construct_classification_input_ref_pair,construct_attention_mask,summarize_attributions,squad_pos_forward_func,construct_input_ref_pair

        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        input_text = text.decode('utf-8')
        max_length = self.setup_config["max_length"]
        model_embedding_class = self.setup_config["model_embedding_class"]
        if self.captum_target_class:
            target = int(self.captum_target_class)
        else:
            target = None
        embed = getattr(self.model, model_embedding_class)
        embedding = embed.embeddings
        logger.info("Received text: '%s'", input_text)
        #preprocessing text for sequence_classification and token_classification.
        if self.setup_config["mode"]== "sequence_classification" or self.setup_config["mode"]== "token_classification" :
            inputs = self.tokenizer.encode_plus(input_text,max_length = int(max_length),pad_to_max_length = True, add_special_tokens = True, return_tensors = 'pt')
            if self.captum_visualization:
                input_ids, ref_input_ids = construct_classification_input_ref_pair(self.tokenizer, input_text, self.ref_token_id, self.sep_token_id, self.cls_token_id)
                attention_mask = construct_attention_mask(input_ids)
                lig = LayerIntegratedGradients(squad_pos_forward_func, embedding)
                attributions_ig, delta = lig.attribute(inputs=input_ids, baselines=ref_input_ids,
                                           additional_forward_args=(self.model,attention_mask, 0),
                                           target=target,
                                           return_convergence_delta=True)

                attributions_sequence_cls_sum = summarize_attributions(attributions_ig)
                print("***********", attributions_sequence_cls_sum)

        #preprocessing text for question_answering.
        elif self.setup_config["mode"]== "question_answering":
            #TODO Reading the context from a pickeled file or other fromats that
            # fits the requirements of the task in hand. If this is done then need to
            # modify the following preprocessing accordingly.

            # the sample text for question_answering in the current version
            # should be formated as dictionary with question and text as keys
            # and related text as values.
            # we use this format here seperate question and text for encoding.

            question_context= ast.literal_eval(input_text)
            question = question_context["question"]
            context = question_context["context"]
            all_input =question+context
            if self.captum_visualization:
                input_ids, ref_input_ids, sep_id = construct_input_ref_pair(self.tokenizer,question, context, self.ref_token_id, self.sep_token_id, self.cls_token_id)
                # ref_input_new = self.construct_qa_input_ref_pair(question,context, self.ref_token_id, self.sep_token_id, self.cls_token_id)
                # input_ids = inputs['input_ids']
                # attention_mask = inputs['attention_mask']
                # ref_ids = self.pad_sequence(ref_input_new,max_length)
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

            inputs = self.tokenizer.encode_plus(question, context,max_length = int(max_length),pad_to_max_length = True, add_special_tokens=True, return_tensors="pt")
        return inputs

    def inference(self, inputs):
        """ Predict the class (or classes) of the received text using the serialized transformers checkpoint.
        """


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

        return [prediction]

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

        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e
