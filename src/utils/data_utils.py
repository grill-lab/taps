import os
import re
import json
import pandas as pd
import pprint
from typing import Union
from copy import deepcopy


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # drop empty columns
    columns_to_drop = df.describe().columns.tolist()
    df = df.drop(columns=columns_to_drop)

    # unify active_intent column values
    df["active_intent"] = df["active_intent"].apply(
        lambda x: [x] if type(x) is str else x
    )

    return df


def load_dataset(fpath: str, preprocess: bool = True) -> pd.DataFrame:
    data = pd.read_json(fpath, lines=True)
    if preprocess:
        data = preprocess_data(data)
    return data


def load_json(fpath: str) -> dict:
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def convert_to_json(example: dict, user_profile: list, add_indent: bool = True) -> str:
    formatted_example = {}
    formatted_example["user_profile"] = user_profile
    dialogue = []
    for line in example["user_utterance"].strip().splitlines():
        role, utterance = line.split(": ", 1)
        dialogue.append(
            {
                "role": "assistant" if role == "Agent" else "user",
                "content": utterance.strip(),
            }
        )
    formatted_example["user_query"] = dialogue
    return pprint.pformat(formatted_example) if add_indent else str(formatted_example)


def load_prompt_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        prompt_text = f.read()
    return prompt_text


def user_profile_to_string(instructions: list) -> str:
    instructions = "> " + "\n> ".join(i for i in instructions)
    return instructions


def get_user_profile(example: pd.Series, return_str: bool = True) -> str:
    user_profile = [
        i["nl_instruction"] for i in example["applicable_standing_instructions"]
    ]
    if return_str:
        user_profile_to_string(instructions=user_profile)
    return user_profile


def get_target(
    example: dict,
) -> str:
    target = str(example["api_calls"]).strip()
    return target


def save_results(data: Union[list, dict], output_dir: str, output_fname: str):
    output_fpath = os.path.join(output_dir, output_fname)
    with open(output_fpath, "w", encoding="utf-8") as f:
        json.dump(data, fp=f, ensure_ascii=False, indent=4)


def augment_instructions(instructions: list, aug_dict: dict):
    aug_instructions = []
    instructions = deepcopy(instructions)
    for instr in instructions:
        if isinstance(instr, str):
            instr = aug_dict[instr]
        else:
            instr["nl_instruction"] = aug_dict[instr["nl_instruction"]]
        aug_instructions.append(instr)
    return aug_instructions


def augment_utterance(user_utterance: str, aug_dict: dict):
    aug_utterance = aug_dict.get(user_utterance)
    return aug_utterance


def load_setup_file(fpath: str) -> tuple:
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)

    shots = data["prog"]["demos"]
    prompt = data["prog"]["signature"]["instructions"]

    shots = pd.DataFrame(shots)
    shots["applicable_standing_instructions"] = shots["user_profile"].apply(
        lambda x: [
            (
                {
                    "nl_instruction": re.sub("^>", "", inst),
                    "instruction_id": i,
                    "instruction_source": "fpath",
                }
                if inst.strip("> ") != ""
                else None
            )
            for (i, inst) in enumerate(x.splitlines())
        ]
    )
    shots["applicable_standing_instructions"] = shots[
        "applicable_standing_instructions"
    ].apply(lambda x: x if x[0] is not None else [])

    shots["api_calls"] = (
        shots["answer"].apply(lambda x: x.replace("---", "")).apply(str.splitlines)
    )
    shots["example_id"] = [
        fpath.split("/")[-1] + "_" + str(i) for i in range(len(shots))
    ]
    return shots, prompt


def format_tag(text: str) -> str:
    text = re.sub(
        "\[SL:([A-Z_]+?) (.+?)\]", "<sl:\g<1>> \g<2></sl>", text, flags=re.MULTILINE
    )
    text = re.sub("\[IN:([A-Z_]+?) ", "<a:\g<1>> ", text, flags=re.MULTILINE)
    text = text.replace("]", "</a>")
    return text


def get_aug_dict(instructions_df: pd.DataFrame, reformat_tag: bool = False) -> dict:
    if reformat_tag:
        instructions_df["aug"] = instructions_df["aug"].apply(format_tag)
    aug_dict = dict(zip(instructions_df["org"], instructions_df["aug"]))
    return aug_dict


def load_tags(fpath: str, new_format: bool = True) -> dict:
    tagged_instructions = pd.read_csv(fpath)
    instr2tag = get_aug_dict(tagged_instructions, reformat_tag=new_format)
    return instr2tag


def get_api_schema(prompt):
    if "Schema" in prompt:
        prompt, api_schema = re.split(
            "(?:\*\*)?Schema:(?:\*\*)?", prompt, flags=re.MULTILINE, maxsplit=1
        )
        api_schema = api_schema.strip()
        prompt = prompt.strip()
        return prompt, api_schema
    else:
        return prompt
