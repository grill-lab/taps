# bot utils
import telegram
import asyncio

import os
import pandas as pd
from typing import Union
from datasets import Dataset
from dotenv import load_dotenv
from pathlib import Path

from evaluation.evaluator import Evaluation
from evaluation.utils import postprocess_preds, get_api_data
from utils.data_utils import (
    preprocess_data,
)

from models import EndpointModel, OpenAIModel, HuggingfaceModel, NLSIModel

from abc import abstractmethod


class NLSIExperiment(object):
    load_dotenv(dotenv_path=Path("../secrets.env"))

    OPENAI_KEY = os.environ["OPENAI_KEY"] if "OPENAI_KEY" in os.environ else None
    ENDPOINT_URL = os.environ["ENDPOINT_URL"] if "ENDPOINT_URL" in os.environ else None

    def __init__(self, args):
        self.args = args

    @staticmethod
    def evaluate(predictions: Union[str, list], test: pd.DataFrame):

        if isinstance(predictions[0], list):
            predictions = list(map(lambda x: "\n".join(x), predictions))

        predictions = postprocess_preds(predictions)

        target_slots, pred_slots, example_ids, _ = get_api_data(
            Dataset.from_pandas(test),
            predictions=predictions,
            alt_names_path="../data/locations.json",
        )

        evals = Evaluation(
            ground_truth=target_slots,
            predictions=pred_slots,
            datum_ids=example_ids,
            preprocess=True,
        )

        agg_m, instance_m = evals.get_metrics()
        return agg_m, instance_m

    def load_model(
        self,
    ) -> NLSIModel:
        # load model and tokenizer
        if self.args.use_gpt:
            if NLSIExperiment.OPENAI_KEY is None:
                raise ValueError("OpenAI key is required for GPT models")
            model = OpenAIModel(
                model_name=self.args.model_name, api_key=NLSIExperiment.OPENAI_KEY
            )
        elif self.args.use_endpoint:
            if NLSIExperiment.ENDPOINT_URL is None:
                raise ValueError("Endpoint URL is required for Endpoint models")
            model = EndpointModel(
                model_name=self.args.model_name,
                endpoint_url=NLSIExperiment.ENDPOINT_URL,
            )
        else:
            model = HuggingfaceModel(
                model_name=self.args.model_name, load_in_4bit=self.args.load_in_4bit
            )
        return model

    @staticmethod
    def load_df(fpath: str) -> pd.DataFrame:
        df = pd.read_json(fpath, lines=True)
        df = preprocess_data(df)
        return df

    @abstractmethod
    def prepare_data(self):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError
