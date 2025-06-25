import os
import pandas as pd
import numpy as np

from utils.data_utils import (
    load_json,
    load_prompt_file,
    load_setup_file,
    get_aug_dict,
)
from experiments.experiment import NLSIExperiment
from utils.constants import (
    DSPY_PROMPT_SUFFIX,
    DSPY_PROMPT_SUFFIX_SIMPLE_TAG,
    DSPY_PROMPT_SUFFIX_TAG,
)


class BaseExperiment(NLSIExperiment):
    def __init__(self, args):
        super().__init__(args=args)

    def prepare_instructions(self) -> tuple:

        # format tagged instructions, if using tagging
        if self.args.tag is not None and self.args.tag == "cot":
            train_instructions = pd.read_csv("../data/train_instructions_golden.csv")
            train_instructions = get_aug_dict(train_instructions, reformat_tag=True)

            test_instructions = None

        elif self.args.simple_aug == "tag":
            train_instructions = pd.read_csv("../data/train_instructions_golden.csv")
            train_instructions = get_aug_dict(train_instructions, reformat_tag=True)

            test_instructions = pd.read_csv(self.args.tag_file_test)
            test_instructions = get_aug_dict(test_instructions, reformat_tag=True)

        else:
            train_instructions = None
            test_instructions = None
        return test_instructions, train_instructions

    def load_prompt(self) -> str:
        print(f"\nLoading prompt from: {self.args.prompt}...")
        prompt = load_prompt_file(self.args.prompt)
        return prompt

    def prepare_from_file(self) -> tuple:
        print(f"\nLoading the setup from file: {self.args.setup_fpath}...")
        train, prompt = load_setup_file(self.args.setup_fpath)

        if self.args.use_setup_shots:
            if self.args.skip_first_demo:
                train = train[1:]
            self.args.n_shots = len(train)

        else:
            train = self.load_df("../data/train.jsonl")

        if not self.args.use_setup_prompt:
            prompt = self.load_prompt()
        else:
            self.args.prompt = self.args.setup_fpath

        prompt_suffix = (
            DSPY_PROMPT_SUFFIX_SIMPLE_TAG
            if self.args.simple_aug == "tag"
            else (
                DSPY_PROMPT_SUFFIX_TAG
                if self.args.tag is not None and self.args.tag == "cot"
                else DSPY_PROMPT_SUFFIX
            )
        )

        if (
            "llama" in self.args.model_name.lower()
            or "mistral-large" in self.args.model_name.lower()
        ):
            prompt_suffix = prompt_suffix.replace(
                "Follow the following format.",
                "The examples are formatted as follows.",
            )
            prompt_suffix = prompt_suffix.strip()
            prompt_end = "You are given several independent examples of the task:"

            prompt_suffix += f"\n\n{prompt_end}"
            prompt_suffix = "\n\n\n" + prompt_suffix + "\n\n\n"

        prompt += prompt_suffix

        return train, prompt

    def prepare_data(self) -> tuple:

        # load evaluation data
        test = self.load_df(self.args.test_dataset)

        # load prompt and ICL demonstrations
        if self.args.setup_fpath is not None:
            train, prompt = self.prepare_from_file()

        else:
            train = self.load_df(self.args.train_dataset)
            prompt = self.load_prompt()

        test_instructions, train_instructions = self.prepare_instructions()

        return test, train, prompt, test_instructions, train_instructions

    def get_output_fpath(self):

        model_name = self.args.model_name.replace("/", "_")
        test_split = self.args.test_dataset.split("/")[-1].split(".")[0]
        aug = (
            self.args.tag
            if self.args.tag is not None
            else f"simple_{self.args.simple_aug}" if self.args.simple_aug else "org"
        )
        res_dir = f"{self.args.output_dir}/{test_split}/{model_name}/{aug}/"
        os.makedirs(res_dir, exist_ok=True)

        fname = (
            model_name
            + ("__endpoint__" if self.args.use_endpoint else "__")
            + self.args.inference_type
            + f"__{aug}"
            + "__"
            + self.args.prompt.split("/")[-1].replace(".txt", "")
            + "__"
            + f"{self.args.n_shots}_shots__"
            + self.args.suffix
            + ("__" if self.args.suffix else "")
            + self.args.test_dataset.split("/")[-1].split(".")[0]
        )

        previous_exps = list(filter(lambda x: x.startswith(fname), os.listdir(res_dir)))
        ind = len(previous_exps)

        fpath = os.path.join(res_dir, fname + f"_{ind}.json")
        return fpath

    def run(self) -> pd.DataFrame:
        model = self.load_model()
        test, train, prompt, test_instructions, train_instructions = self.prepare_data()
        predictions = model.get_model_predictions(
            test=test,
            train=train,
            prompt=prompt,
            tag=self.args.tag,
            simple_aug=self.args.simple_aug,
            dspy=self.args.setup_fpath is not None,
            test_instructions=test_instructions,
            train_instructions=train_instructions,
            n_shots=self.args.n_shots,
            max_tokens=self.args.max_tokens,
            do_sample=self.args.do_sample,
            num_beams=self.args.num_beams,
            temperature=self.args.temperature,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            seed=self.args.seed,
            inference_type=self.args.inference_type,
            stop_strings=["\n\n\n\n", "---", "\nDialogue:"],
            max_samples=self.args.max_samples,
        )

        # save experiment arguments
        predictions["args"] = [self.args.__dict__] * predictions.shape[0]

        # get metrics
        agg_metrics, instance_metrics = self.evaluate(
            predictions=predictions["reference_api_calls"].tolist(), test=test
        )
        predictions["em_score"] = instance_metrics["em_scores"]
        predictions["f1_score"] = instance_metrics["f1_scores"]
        predictions["prec_score"] = instance_metrics["prec_scores"]
        predictions["rec_score"] = instance_metrics["rec_scores"]
        
        predictions["avg_score"] = np.mean(
            [
                instance_metrics["em_scores"],
                instance_metrics["f1_scores"],
                instance_metrics["prec_scores"],
                instance_metrics["rec_scores"],
            ],
            axis=0,
        )

        agg_metrics["avg_score"] = np.mean(
            [
                agg_metrics["em_score"],
                agg_metrics["f1_score"],
                agg_metrics["prec_score"],
                agg_metrics["rec_score"],
            ]
        )

        predictions["agg_metrics"] = [agg_metrics] * predictions.shape[0]

        # save evaluation results
        fpath = self.get_output_fpath()
        print("\nSaving to file", fpath)

        predictions.to_json(
            fpath,
            orient="records",
            indent=4,
            force_ascii=False,
        )
        return predictions
