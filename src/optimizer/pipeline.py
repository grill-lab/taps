import os
import random
import pandas as pd
import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

from utils.arguments import GenerationParameters

from optimizer.data_formatter import NLSIDataFormatter
from optimizer.signatures import (
    NLSISignature,
    NLSISignatureAPI,
    NLSISignatureCoTTag,
    NLSISignatureSimpleTag,
)
from optimizer.nlsi_client import NLSIClient
from optimizer.nlsi_model import NLSIModel
from optimizer.metrics import exact_match, f1

from typing import Optional


class DSPyPipeline(object):
    def __init__(
        self,
        model_name: str,
        gen_params: GenerationParameters,
        prompt_file: str = "../prompts/default_prompt.txt",
        tag: Optional[str] = None,
        api_in_prompt: bool = True,
        metric: str = "em",
        num_threads: int = 1,
        max_bootstrapped_demos: int = 5,
        max_labeled_demos: int = 5,
        num_candidate_programs: int = 10,
        output_dir: str = "../results/dspy/",
        load_in_4bit: bool = False,
        seed: int = 42,
    ):
        self.model_name = model_name
        self.gen_params = gen_params
        self.prompt_file = prompt_file
        self.tag = tag
        self.api_in_prompt = api_in_prompt
        self.metric_name = metric
        self.metric = exact_match if self.metric_name == "em" else f1
        self.num_threads = num_threads
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.num_candidate_programs = num_candidate_programs
        self.output_dir = output_dir
        self.load_in_4bit = load_in_4bit
        self.seed = seed
        self.api_key = os.environ["OPENAI_KEY"] if "OPENAI_KEY" in os.environ else None

        self.load()

    def load(self):
        self.data_formatter = self.load_data_formatter()
        self.signature = self.get_signature()
        self.nlsi_client = self.load_client()
        self.nlsi_model = self.load_model()
        self.optimizer = self.load_optimizer()

    def load_data_formatter(self) -> NLSIDataFormatter:
        return NLSIDataFormatter(
            prompt_file=self.prompt_file, tag=self.tag, api_in_prompt=self.api_in_prompt
        )

    def get_evaluator(self, devset: list) -> Evaluate:
        evaluate = Evaluate(
            devset=devset,
            metric=self.metric,
            num_threads=self.num_threads,
            display_progress=True,
            display_table=False,
            provide_traceback=True,
        )
        return evaluate

    def get_signature(self) -> dspy.Signature:
        if not self.api_in_prompt:
            signature = NLSISignatureAPI

        if self.tag is None:
            signature = NLSISignature
        elif self.tag == "simple":
            signature = NLSISignatureSimpleTag
        elif self.tag == "cot":
            signature = NLSISignatureCoTTag

        signature.instructions = self.data_formatter.prompt + (
            ""
            if not self.api_in_prompt
            else "\n\nSchema:\n" + self.data_formatter.api_schema
        )
        return signature

    def update_signature(self, prompt: str) -> None:
        self.signature.instructions = prompt

    def load_client(self) -> NLSIClient:
        nlsi_client = NLSIClient(
            model_name=self.model_name,
            api_schema=self.data_formatter.api_schema,
            gen_params=self.gen_params,
            load_in_4bit=self.load_in_4bit,
            api_key=self.api_key,
        )
        dspy.configure(lm=nlsi_client)
        return nlsi_client

    def load_model(self) -> NLSIModel:
        return NLSIModel(tag=self.tag, api_in_prompt=self.api_in_prompt)

    def load_optimizer(self) -> BootstrapFewShotWithRandomSearch:
        optimizer = BootstrapFewShotWithRandomSearch(
            max_bootstrapped_demos=self.max_bootstrapped_demos,
            max_labeled_demos=self.max_labeled_demos,
            num_candidate_programs=self.num_candidate_programs,
            num_threads=self.num_threads,
            metric=self.metric,
            teacher_settings=dict(lm=self.nlsi_client),
        )
        return optimizer

    def get_response(
        self,
        user_utterance: str,
        user_profile: str,
        api_schema: Optional[str] = None,
        tagged_user_profile: Optional[str] = None,
    ) -> str:
        response = self.nlsi_model(
            api_schema=(
                api_schema if api_schema is not None else self.data_formatter.api_schema
            ),
            user_utterance=user_utterance,
            user_profile=user_profile,
            tagged_user_profile=tagged_user_profile,
        )
        return response.answer

    def save_program(self, program: dspy.Predict, score: float, history: list):
        mname = self.model_name.replace("/", "_")
        output_dir = os.path.join(self.output_dir, mname)
        os.makedirs(output_dir, exist_ok=True)

        print(f"Saving program to {output_dir}...")
        fname = (
            f"bootstrap_fewshot_{self.prompt_file.split('/')[-1].replace('.txt', '')}_"
            + (self.tag if self.tag is not None else "")
            + f"_{self.metric_name}_{round(score, 2)}.json"
        )

        prog_path = f"{output_dir}/{fname.rsplit('.',1)[0].replace('.', ',')}_prog/"
        program.save(f"{output_dir}/{fname}")
        program.save(
            prog_path,
            save_program=True,
        )

        history = pd.DataFrame(history)
        history.to_csv(
            f"{prog_path}/preds.csv",
            index=False,
        )

    def run(
        self,
        test_fpath: str = "../data/val.jsonl",
        train_fpath: str = "../data/train.jsonl",
        train_profile_tags: Optional[str] = "../data/train_instructions_golden.csv",
        test_profile_tags: Optional[str] = "../data/gpt4_tag_2_shot.csv",
        max_samples: int = -1,
        max_samples_eval: int = -1,
    ):
        # load data
        trainset = self.data_formatter(
            train_fpath, tags_file=train_profile_tags if self.tag is not None else None
        )
        devset = self.data_formatter(
            test_fpath, tags_file=test_profile_tags if self.tag is not None else None
        )
        print(devset[0], end="\n\n\n")
        print(self.nlsi_model.prog, end="\n\n\n")

        # load evaluator
        evaluate = self.get_evaluator(
            devset=devset[:max_samples_eval] if max_samples_eval > -1 else devset
        )

        # optimization
        random.seed(self.seed)
        optimized_prog = self.optimizer.compile(
            self.nlsi_model,
            trainset=trainset,
            valset=(
                random.sample(devset, k=max_samples) if max_samples > -1 else devset
            ),
        )

        self.nlsi_client.history = []

        # evaluation
        score = evaluate(optimized_prog, metric=self.metric)
        print(f"FINAL SCORE: {self.metric_name}={score}")

        self.save_program(
            program=optimized_prog, score=score, history=self.nlsi_client.history
        )

        return optimized_prog
