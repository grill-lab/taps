import pandas as pd

from abc import abstractmethod
from tqdm.auto import tqdm
from typing import Union, Optional

from utils.prompt_utils import format_input


class NLSIModel(object):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.load()

    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        formatted_example: str,
        max_tokens: int = 300,
        do_sample: bool = False,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
        seed: int = 42,
        inference_type: str = "chat_completion",
        stop_strings: list = ["\n\n\n\n", "---", "\nDialogue:"],
    ) -> str:
        raise NotImplementedError

    def get_model_predictions(
        self,
        test: pd.DataFrame,
        train: pd.DataFrame,
        prompt: str,
        tag: bool,
        simple_aug: Optional[str] = None,
        use_system_prompt: bool = False,
        dspy: bool = False,
        test_instructions: Optional[dict] = None,
        train_instructions: Optional[dict] = None,
        n_shots: int = 1,
        max_tokens: int = 300,
        do_sample: bool = False,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
        seed: int = 42,
        inference_type: str = "chat_completion",
        stop_strings: list = ["\n\n\n\n", "---"],
        max_samples: int = -1,
        output_scores: bool = False,
    ) -> pd.DataFrame:

        print("Starting evaluation...")

        if max_samples > 0:
            test = test.iloc[:max_samples]

        predictions = []
        for ind, example in tqdm(test.iterrows(), total=test.shape[0]):

            if len(train) != n_shots:
                demonstrations = train.sample(n_shots, random_state=ind + seed)
            else:
                demonstrations = train

            formatted_example = format_input(
                example=example,
                prompt=prompt,
                demonstrations=demonstrations,
                tag=tag,
                simple_aug=simple_aug,
                dspy=dspy,
                test_instructions=test_instructions,
                train_instructions=train_instructions,
                eos_token=(
                    self.tokenizer.eos_token if self.tokenizer is not None else "</s>"
                ),
            )

            if (
                "llama" in self.model_name.lower()
                or "mistral-large" in self.model_name.lower()
            ):
                if tag is not None:
                    if tag == "cot":
                        formatted_example = "---\n\nGiven the examples above, output only the tagged instructions followed by the API calls for the following example with no additional text:".join(
                            formatted_example.rsplit("---", 1)
                        )
                    if tag == "cot_dials":
                        formatted_example = "---\n\nGiven the examples above, output only the tagged dialogue, tagged instructions and the final API calls for the following example with no additional text:".join(
                            formatted_example.rsplit("---", 1)
                        )
                else:
                    formatted_example = "---\n\nGiven the examples above, output only the API calls for the following example with no additional text:".join(
                        formatted_example.rsplit("---", 1)
                    )

            if ind == 0:
                print("\n\n[INPUT]\n" + formatted_example)

            model_pred = self.predict(
                formatted_example=formatted_example,
                use_system_prompt=use_system_prompt,
                max_tokens=max_tokens,
                do_sample=do_sample,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=seed,
                inference_type=inference_type,
                stop_strings=stop_strings,
                output_scores=output_scores,
            )

            if output_scores:
                model_pred, score = model_pred

            if ind == 0:
                print("\n\n[OUTPUT]\n" + model_pred)

            row = example.to_dict()
            row["n_shots"] = n_shots
            row["shot_ids"] = demonstrations["example_id"].tolist()
            row["reference_api_calls"] = model_pred
            row["input"] = formatted_example

            if output_scores:
                row["score"] = score

            predictions.append(row)

        return pd.DataFrame(predictions)
