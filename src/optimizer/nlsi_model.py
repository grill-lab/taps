import dspy
from optimizer.signatures import (
    NLSISignature,
    NLSISignatureAPI,
    NLSISignatureCoTTag,
    NLSISignatureSimpleTag
)

from typing import Optional


class NLSIModel(dspy.Module):
    def __init__(self, tag: Optional[str] = None, api_in_prompt: bool = True):
        print("Initializing CoT for NLSI task...")
        super().__init__()
        self.tag = tag
        self.api_in_prompt = api_in_prompt

        if not self.api_in_prompt:
            self.prog = dspy.Predict(NLSISignatureAPI)

        if tag is None:
            self.prog = dspy.Predict(NLSISignature)
        elif tag == "simple":
            self.prog = dspy.Predict(NLSISignatureSimpleTag)
        elif tag == "cot":
            self.prog = dspy.Predict(NLSISignatureCoTTag)
        else:
            raise ValueError(
                "`tag` can take the value of [None, 'simple', 'cot'] bit given %s" % tag
            )

    def forward(
        self,
        user_utterance,
        user_profile,
        api_schema: Optional[str] = None,
        tagged_user_profile: Optional[str] = None,
        tagged_user_utterance: Optional[str] = None,
    ):
        config = {"api_schema": api_schema}

        if self.prog.signature.__name__ == "NLSISignatureAPI":
            out = self.prog(
                api_schema=api_schema,
                user_utterance=user_utterance,
                user_profile=user_profile,
                tagged_user_profile=tagged_user_profile,
                config=config,
            )
        elif self.prog.signature.__name__ == "NLSISignatureCoTTag":
            out = self.prog(
                user_utterance=user_utterance,
                user_profile=user_profile,
                config=config,
            )
        else:
            out = self.prog(
                user_utterance=user_utterance,
                user_profile=user_profile,
                tagged_user_profile=tagged_user_profile,
                config=config,
            )

        return out
