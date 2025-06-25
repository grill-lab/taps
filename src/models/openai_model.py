import numpy as np
from openai import OpenAI

from models.model import NLSIModel


class OpenAIModel(NLSIModel):
    def __init__(self, model_name: str, api_key: str):
        self.api_key = api_key
        super().__init__(model_name=model_name)

    def load(self):
        self.model = OpenAI(api_key=self.api_key)
        self.tokenizer = None

    def predict(
        self,
        formatted_example: str,
        use_system_prompt: bool = False,
        max_tokens: int = 300,
        do_sample: bool = False,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        seed: int = 42,
        inference_type: str = "chat_completion",
        stop_strings: list = ["\n\n\n\n", "---", "\nDialogue:"],
        output_scores: bool = False,
    ):

        out = self.model.chat.completions.create(
            messages=[{"role": "user", "content": formatted_example}],
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=True,
            seed=seed,
        )

        pred = out.choices[0].message.content
        pred = pred.split("<EOS>")[0]

        if output_scores:

            logprobs = np.array(
                [token.logprob for token in out.choices[0].logprobs.content]
            )
            score = logprobs.sum() / logprobs.shape[0]

            return pred.strip(), score

        return pred.strip()
