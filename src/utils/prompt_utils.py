import pandas as pd

from typing import Optional


def format_test_example(
    example: pd.Series,
    tag: Optional[str] = None,
    simple_aug: Optional[dict] = None,
    instructions: Optional[dict] = None,
) -> str:
    """
    =====================================  =========================================  ========================================
    ORIGINAL                               TAG SIMPLE                                 TAG AND GENERATE
    =====================================  =========================================  ========================================
    Dialogue:                              Dialogue:                                  Dialogue:
    <user_utterance>                       <user_utterance>                           <user_utterance>

    Applicable Standing Instructions:      Applicable Standing Instructions:          Applicable Standing Instructions:
    > <instruction 1>                      > <instruction 1>                          > <instruction 1>
    > <instruction 2>                      > <instruction 2>                          > <instruction 2>

    API Calls:                             Tagged Applicable Standing Instructions:   Tagged Applicable Standing Instructions:
                                           > <augmented instruction 1>
                                           > <augmented instruction 1>

                                           API Calls:
    =====================================  =========================================  ========================================
    """
    formatted_example = (
        "Dialogue:\n"
        + example["user_utterance"].strip()
        + "\n\nApplicable Standing Instructions:\n> "
    )

    if simple_aug and simple_aug in ["task", "pref"]:
        formatted_example += "\n> ".join(
            [
                instructions[instr["nl_instruction"].strip()]
                for instr in example["applicable_standing_instructions"]
            ]
        )
    else:
        formatted_example += "\n> ".join(
            [
                instr["nl_instruction"].strip()
                for instr in example["applicable_standing_instructions"]
            ]
        )

    if tag is not None or simple_aug == "tag":
        if tag == "cot_dials":
            formatted_example += "\n\nTagged Dialogue:\n"
        else:
            formatted_example += "\n\nTagged Applicable Standing Instructions:\n"
            if simple_aug == "tag":
                if "tagged_user_profile" in example:
                    formatted_example += example["tagged_user_profile"]

                # if we have full tagged profiles
                elif ">" in list(instructions)[0]:
                    formatted_example += instructions[
                        "> "
                        + "\n> ".join(
                            [
                                i["nl_instruction"]
                                for i in example["applicable_standing_instructions"]
                            ]
                        )
                    ]
                else:
                    formatted_example += "> " + "\n> ".join(
                        [
                            instructions[instr["nl_instruction"].strip()]
                            for instr in example["applicable_standing_instructions"]
                        ]
                    )
                formatted_example += "\n\nAPI Calls:\n"
    else:
        formatted_example += "\n\nAPI Calls:\n"

    return formatted_example


def format_train_example(
    example: pd.Series,
    tag: Optional[str] = None,
    simple_aug: Optional[dict] = None,
    train_instructions: Optional[dict] = None,
) -> str:

    formatted_example = format_test_example(
        example, tag=tag, simple_aug=simple_aug, instructions=train_instructions
    )

    if tag is not None:
        if tag == "cot_dials":
            if "tagged_user_utterance" in example:
                formatted_example += example["tagged_user_utterance"]
            else:
                formatted_example += train_instructions[example["user_utterance"]]
            formatted_example += "\n\nTagged Applicable Standing Instructions:\n"

        if "tagged_user_profile" in example:
            formatted_example += example["tagged_user_profile"] + "\n\nAPI Calls:\n"
        else:
            formatted_example += (
                "> "
                + "\n> ".join(
                    [
                        train_instructions[instr["nl_instruction"]].strip()
                        for instr in example["applicable_standing_instructions"]
                    ]
                )
                + "\n\nAPI Calls:\n"
            )

    if "answer" in example:
        formatted_example += example["answer"]
    else:
        formatted_example += "\n".join(example["api_calls"])
    return formatted_example


def format_input(
    example: pd.Series,
    prompt: str,
    demonstrations: list,
    tag: Optional[str] = None,
    simple_aug: Optional[None] = None,
    dspy: bool = False,
    train_instructions: Optional[dict] = None,
    test_instructions: Optional[dict] = None,
    eos_token: str = "</s>",
) -> str:

    if ("<sl" in prompt) is not (tag is not None or simple_aug == "tag"):
        raise ValueError(
            "Passing the incorrect prompt: prompt and tagging approach do not match."
        )
    if simple_aug and tag is not None:
        raise ValueError("Cannot do both CoT tagging and simple augs")

    if "{{message}}" in prompt:
        formatted_input = (
            "\n\n\n Here are some examples:\n\n" if len(demonstrations) > 0 else ""
        )
    else:
        formatted_input = prompt.strip() + "\n\n\n"

    for demonstration in demonstrations.to_dict(orient="records"):
        formatted_input += format_train_example(
            demonstration,
            tag=tag,
            simple_aug=simple_aug,
            train_instructions=train_instructions,
        )
        if dspy:
            formatted_input += "\n\n---\n\n"
        else:
            # formatted_input += "\n" + eos_token + "\n\n\n"
            formatted_input += "\n\n\n"

    formatted_test_example = format_test_example(
        example, tag=tag, simple_aug=simple_aug, instructions=test_instructions
    )
    if "{{message}}" in prompt:
        if tag is not None:
            formatted_test_example = formatted_test_example.split(
                "Tagged Applicable Standing Instructions:"
            )[0].strip()
        prompt = prompt.replace("{{message}}", formatted_test_example)
        if len(demonstrations) > 0:
            formatted_input = prompt.replace("{{examples}}", formatted_input.strip())

    else:
        formatted_input += formatted_test_example
    return formatted_input
