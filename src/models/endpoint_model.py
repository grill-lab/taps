from huggingface_hub import InferenceClient
from transformers import AutoTokenizer

from models.model import NLSIModel


class EndpointModel(NLSIModel):
    def __init__(self, model_name: str, endpoint_url: str):
        self.endpoint_url = endpoint_url
        super().__init__(model_name=model_name)

    def load(self):
        self.model = InferenceClient(self.endpoint_url)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

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
        num_return_sequences: int = 1,
        seed: int = 42,
        inference_type: str = "chat_completion",
        stop_strings: list = ["\n\n\n\n", "---", "\nDialogue:"],
        output_scores: bool = False,
    ) -> str:

        if inference_type == "chat_completion":
            if "OLMo" in self.model_name and use_system_prompt:
                system_prompt = "You are OLMo 2, a helpful and harmless AI Assistant built by the Allen Institute for AI."
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": formatted_example},
                ]
            else:
                messages = [{"role": "user", "content": formatted_example}]

            out = self.model.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
            pred = out.choices[0].message.content

        elif inference_type == "text_generation":
            pred = self.model.text_generation(
                prompt=formatted_example,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=seed,
                stop_sequences=[
                    "</s>" if self.tokenizer is None else self.tokenizer.eos_token
                ]
                + stop_strings,
            )

        return pred.strip()
