import dspy
import pandas as pd
from utils.data_utils import (
    load_dataset,
    load_tags,
    load_prompt_file,
    get_api_schema,
    get_user_profile,
    user_profile_to_string,
    augment_instructions,
    augment_utterance,
)

from typing import Optional


class NLSIDataFormatter(object):
    def __init__(
        self,
        prompt_file: str,
        tag: Optional[str] = None,
        api_in_prompt: bool = True,
    ) -> None:
        """Formatting data for DSPy optimizer

        Args:
            prompt_file (str): path to he prompt file.
            tag (Optional[str], optional): tagging strategy for standing instructions. Defaults to None. Possible values:
                None - do not include tagged instructions
                `simple` - include tags in the prompt
                `cot` - tags as part of the output
            api_in_prompt (bool, optional): whether to include API Schema in the prompt or each example. Defaults to True.

        Raises:
            ValueError: _description_
        """
        self.tag = tag
        if tag not in [None, "simple", "cot", "cot_dials"]:
            raise ValueError(
                "`tag` can take the value of [None, 'simple', 'cot', 'cot_dials'] bit given %s"
                % tag
            )
        self.api_in_prompt = api_in_prompt
        self.prompt, self.api_schema = NLSIDataFormatter.load_prompt_and_api(
            prompt_file
        )
        self.input_fields = self.get_input_fields()

    @staticmethod
    def load_prompt_and_api(fpath: Optional[str] = None) -> Optional[str]:
        if fpath is not None:
            prompt = load_prompt_file(fpath)
            prompt, api_schema = get_api_schema(prompt)
        else:
            prompt, api_schema = None, None
        return prompt, api_schema

    def get_input_fields(self) -> list:
        input_fields = ["user_profile", "user_utterance"]
        if not self.api_in_prompt:
            input_fields = ["api_schema"] + input_fields
        if self.tag:
            if self.tag == "cot_dials":
                input_fields += ["tagged_user_utterance"]
            input_fields += ["tagged_user_profile"]
        return input_fields

    def format_example(
        self, example: pd.Series, aug_dict: Optional[dict] = None
    ) -> dspy.Example:

        user_profile = get_user_profile(example, return_str=False)

        if self.tag:
            if self.tag == "simple" and list(aug_dict)[0].startswith(">"):
                tagged_user_profile = aug_dict[
                    "> "
                    + "\n> ".join(
                        [
                            i["nl_instruction"]
                            for i in example["applicable_standing_instructions"]
                        ]
                    )
                ]

            else:
                if self.tag == "cot_dials":
                    tagged_user_query = augment_utterance(
                        example["user_utterance"], aug_dict=aug_dict
                    )

                tagged_user_profile = augment_instructions(
                    user_profile, aug_dict=aug_dict
                )
                tagged_user_profile = user_profile_to_string(tagged_user_profile)

        user_profile = user_profile_to_string(user_profile)
        user_query = example["user_utterance"]

        answer = "\n".join(example["api_calls"])

        formatted_example = {
            "user_profile": user_profile,
            "user_utterance": user_query,
            "answer": answer,
        }

        if not self.api_in_prompt:
            formatted_example["api_schema"] = self.api_schema

        # if self.tag == "simple":
        if self.tag:
            if self.tag == "cot_dials":
                formatted_example["tagged_user_utterance"] = tagged_user_query
            formatted_example["tagged_user_profile"] = tagged_user_profile

        formatted_example = dspy.Example(formatted_example).with_inputs(
            *self.input_fields
        )
        return formatted_example

    def format_dataset(
        self, dataset: pd.DataFrame, aug_dict: Optional[dict] = None
    ) -> list:
        formatted_data = []
        for _, example in dataset.iterrows():
            formatted_data.append(
                self.format_example(example=example, aug_dict=aug_dict)
            )
        return formatted_data

    def load_data(self, fpath: str, tags_file: Optional[str] = None) -> list:
        if self.tag and tags_file is None:
            raise ValueError(
                "Chosen tagging strategy `%s` but no `tags_file` provided." % self.tag
            )

        # load raw dataset
        dataset = load_dataset(fpath)

        # load tags for instructions if needed
        instr2tag = None
        if self.tag:
            instr2tag = load_tags(tags_file)

        formatted_dataset = self.format_dataset(dataset=dataset, aug_dict=instr2tag)
        return formatted_dataset

    def __call__(self, fpath: str, tags_file: Optional[str] = None) -> list:
        return self.load_data(fpath=fpath, tags_file=tags_file)
