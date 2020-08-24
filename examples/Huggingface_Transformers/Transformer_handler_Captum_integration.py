from abc import ABC
import json
import logging
import os
import ast
import torch
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForQuestionAnswering,AutoModelForTokenClassification,AutoConfig
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
from ts.torch_handler.insight_handler import *

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
        # read configs for the mode, model_name, etc. from setup_config.json
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


    def preprocess(self, requests):
        """ Basic text preprocessing, based on the user's chocie of application mode.
        """
        input_batch = None
        bodies = []
        for idx, data in enumerate(requests):
            text = data.get("data")
            if text is None:
                text = data.get("body")
            input_text = text.decode('utf-8')
            Body = ast.literal_eval(input_text)
            bodies.append(Body)
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
            input_ids = inputs["input_ids"].to(self.device)
            if input_ids.shape is not None:
                if input_batch is None:
                    input_batch = input_ids
                else:
                    input_batch = torch.cat((input_batch, input_ids), 0)
        print(bodies)
        return input_batch, bodies

    def inference(self, input_batch):
        """ Predict the class (or classes) of the received text using the serialized transformers checkpoint.
        """
        inferences = []
        # Handling inference for sequence_classification.
        if self.setup_config["mode"]== "sequence_classification":
            predictions = self.model(input_batch)
            print("This the output size from the Seq classification model", predictions[0].size())
            print("This the output from the Seq classification model", predictions)
            print("WE Got the predictionssssssssssssssssssssssssss")
            num_rows, num_cols = predictions[0].shape
            for i in range(num_rows):
                out = predictions[0][i].unsqueeze(0)
                y_hat= out.argmax(1).item()
                predicted_idx = str(y_hat)
                inferences.append(self.mapping[predicted_idx])
            print("WE received the inferencessssssssssssssssssssssss")

        # Handling inference for question_answering.
        elif self.setup_config["mode"]== "question_answering":
            # the output should be only answer_start and answer_end
            # we are outputing the words just for demonstration.
            answer_start_scores, answer_end_scores = self.model(input_batch)
            print("This the output size for answer start scores from the question answering model", answer_start_scores.size())
            print("This the output for answer start scores from the question answering model", answer_start_scores)
            print("This the output size for answer end scores from the question answering model", answer_end_scores.size())
            print("This the output for answer end scores from the question answering model", answer_end_scores)

            num_rows, num_cols = answer_start_scores.shape
            # inferences = []
            for i in range(num_rows):
                answer_start_scores_one_seq = answer_start_scores[i].unsqueeze(0)
                answer_start= torch.argmax(answer_start_scores_one_seq)
                answer_end_scores_one_seq = answer_end_scores[i].unsqueeze(0)
                answer_end= torch.argmax(answer_end_scores_one_seq) + 1
                prediction = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_batch[i].tolist()[answer_start:answer_end]))
                inferences.append(prediction)
            logger.info("Model predicted: '%s'", prediction)
        # Handling inference for token_classification.
        elif self.setup_config["mode"]== "token_classification":
            outputs = self.model(input_batch)[0]
            print("This the output size from the token classification model", outputs.size())
            print("This the output from the token classification model",outputs)
            num_rows = outputs.shape[0]
            for i in range(num_rows):
                output = outputs[i].unsqueeze(0)
                predictions = torch.argmax(output, dim=2)
                tokens = self.tokenizer.tokenize(self.tokenizer.decode(input_batch[i]))
                if self.mapping:
                    label_list = self.mapping["label_list"]
                label_list = label_list.strip('][').split(', ')
                prediction = [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].tolist())]
                inferences.append(prediction)
            logger.info("Model predicted: '%s'", prediction)

        return inferences

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
        final_reponse = []
        data,bodies = _service.preprocess(data)
        response_list = []
        for Body in bodies:
            if "interperation" in Body and Body["interperation"]==True:
                response = []
                # if _service.setup_config["mode"]== "question_answering":
                #     response["text"] = Body["question"] +" "+ Body["context"]
                # else:
                #     response["text"] = Body["text"]
                insight_handler = Insight(InsightMethod("WordImportance"),_service, Body)
                dom = insight_handler.insight()
                tmpdir = tempfile.mkdtemp()
                filename = 'visualization.html'
                if Body["output"]=="html":
                    path = os.path.join(tmpdir,filename)
                    print(path)
                    try:
                        with open(path, "w") as tmp:
                            for line in dom:
                                tmp.write(line)
                        # response["visualization"] = path
                        response.append(path)
                    except IOError as e:
                        print('IOError')
                elif Body["output"]=="json":
                    # response["visualization"] = dom
                    response.append(dom)
                response_list.append(response)
            else:
                response_list.append("not required")


        data = _service.inference(data)

        data = _service.postprocess(data)

        try:
            if len(response_list)>0:
                for index, item in enumerate(data):

                    final_reponse.append((data[index],response_list[index]))

                return final_reponse
            else:
                return data
        except  Exception as e:
            raise e

        # return data
    except Exception as e:
        raise e
