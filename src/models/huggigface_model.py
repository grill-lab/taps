from models.model import NLSIModel
from utils.model_utils import load_model_and_tokenizer


class HuggingfaceModel(NLSIModel):
    def __init__(self, model_name: str, load_in_4bit: bool = False):
        self.load_in_4bit = load_in_4bit
        super().__init__(model_name=model_name)

    def load(self):
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name=self.model_name, load_in_4bit=self.load_in_4bit
        )

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

            tok_input = {
                "input_ids": self.tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                ).to(self.model.device)
            }

        else:
            tok_input = self.tokenizer(formatted_example, return_tensors="pt").to(
                self.model.device
            )

        out = self.model.generate(
            **tok_input,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
            stop_strings=[
                "</s>" if self.tokenizer is None else self.tokenizer.eos_token
            ]
            + stop_strings,
            tokenizer=self.tokenizer,
            return_dict_in_generate=True,
            output_scores=True,
        )
        pred = self.tokenizer.batch_decode(
            out.sequences[:, tok_input["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        pred = pred[0].strip() if num_return_sequences == 1 else pred

        if output_scores:
            return pred, out.sequences_scores.cpu().numpy().item()
        else:
            return pred
