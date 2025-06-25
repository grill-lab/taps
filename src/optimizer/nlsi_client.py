from openai import OpenAI
from dsp import LM
import re
from utils.arguments import GenerationParameters
from utils.model_utils import load_model_and_tokenizer

from typing import Optional


class NLSIClient(LM):
    def __init__(
        self,
        model_name: str,
        api_schema: str,
        gen_params: GenerationParameters,
        load_in_4bit: bool = False,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name

        if "gpt" in self.model_name:
            self.model = OpenAI(api_key=api_key)
        else:
            self.model, self.tokenizer = load_model_and_tokenizer(
                model_name=model_name, load_in_4bit=load_in_4bit
            )
        self.api_schema = api_schema
        self.load_in_4bit = load_in_4bit
        self.kwargs = gen_params.__dict__
        self.api_key = api_key

        self.history = []

    def basic_request(self, prompt, **kwargs):

        if prompt.count("\nSchema:\n") > 1:
            prompt = re.sub(
                "Schema:\n[^<].+?\n", "Schema:\n" + self.api_schema + "\n", prompt
            )

        # reformat user queries
        prompt = re.sub(" ((User:)|(Agent:))", "\n\g<1>", prompt, flags=re.MULTILINE)

        # reformat user profiles
        prompt = re.sub(
            "[^\n]\nApplicable Standing Instructions:",
            "\n\nApplicable Standing Instructions:",
            prompt,
            flags=re.MULTILINE,
        )
        prompt = re.sub(
            "[^\n]\nTagged Standing Instructions:",
            "\n\nTagged Standing Instructions:",
            prompt,
            flags=re.MULTILINE,
        )
        prompt = re.sub(
            "[^\n]\nTagged Dialogue:",
            "\n\nTagged Dialogue:",
            prompt,
            flags=re.MULTILINE,
        )
        prompt = prompt.replace(" >", "\n>").replace("\n\n>", "\n>")

        # reformat targets
        prompt = re.sub(
            "[^\n]\nAPI Calls:", "\n\nAPI Calls:", prompt, flags=re.MULTILINE
        )
        prompt = re.sub(
            " API Calls: Get", "\n\nAPI Calls: Get", prompt, flags=re.MULTILINE
        )
        prompt, examples = prompt.split("---", 1)
        examples = re.sub(" (Get[A-Z])", "\n\g<1>", examples, flags=re.MULTILINE)
        prompt += examples

        if (
            "llama" in self.model_name
            or "mistral-large" in self.model_name.lower()
            or "gpt" in self.model_name.lower()
        ):
            prompt = prompt.replace(
                "Follow the following format.", "The examples are formatted as follows."
            )

            num_demos = len(re.findall("(^|\n)Dialogue:\n", prompt))
            if num_demos > 2:
                prompt_end = "You are given several independent examples of the task:"

                prompt = re.sub(
                    "(The examples are formatted as follows.+?)---",
                    f"\g<1>---\n\n{prompt_end}\n\n---",
                    prompt,
                    flags=re.DOTALL,
                )

                if (
                    "Tagged Standing Instructions:" not in prompt
                    or "<tagged applicable standing instructions>" in prompt
                ):
                    prompt = "---\n\nGiven the examples above, output only the API calls for the following example with no additional text:".join(
                        prompt.rsplit("---", 1)
                    )
                elif "Tagged Dialogue:" in prompt:
                    prompt = "---\n\nGiven the examples above, output only the tagged dialogue, tagged instructions and the final API calls for the following example with no additional text:".join(
                        prompt.rsplit("---", 1)
                    )
                else:
                    prompt = "---\n\nGiven the examples above, output only the tagged instructions followed by the API calls for the following example with no additional text:".join(
                        prompt.rsplit("---", 1)
                    )
            else:
                prompt_end = (
                    "Output only the API calls with no additional text."
                    if "Tagged Standing Instructions:" not in prompt
                    or "<tagged applicable standing instructions>" in prompt
                    else (
                        "Output only the Tagged Standing Instructions and API calls with no additional text."
                        if "Tagged Dialogue:" not in prompt
                        else "Output only the Tagged Dialogue, Tagged Standing Instructions and API calls with no additional text."
                    )
                )
                prompt = re.sub(
                    "(The examples are formatted as follows.+?)---",
                    f"\g<1>---\n\n{prompt_end}\n\n---",
                    prompt,
                    flags=re.DOTALL,
                )
            if "Tagged Dialogue:" in prompt:
                prompt = prompt.rsplit("Tagged Dialogue:", 1)[0] + "Tagged Dialogue:"
            elif (
                "Tagged Standing Instructions:" in prompt
                and "<tagged applicable standing instructions>" not in prompt
            ):
                prompt = (
                    prompt.rsplit("Tagged Standing Instructions:", 1)[0]
                    + "Tagged Standing Instructions:"
                )

        messages = [{"role": "user", "content": prompt}]

        if "gpt" in self.model_name:
            out = self.model.chat.completions.create(
                messages=messages,
                model=self.model_name,
                max_tokens=self.kwargs.get("max_tokens"),
            )
            formatted_answer = (
                out.choices[0].message.content.split("API Calls:", 1)[-1].strip()
            )

        else:
            inputs = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt"
            ).to(self.model.device)


            out = self.model.generate(
                inputs,
                max_new_tokens=self.kwargs.get("max_new_tokens"),
                do_sample=self.kwargs.get("do_sample"),
                num_beams=self.kwargs.get("num_beams"),
                temperature=self.kwargs.get("temperature"),
                top_k=self.kwargs.get("top_k"),
                top_p=self.kwargs.get("top_p"),
                pad_token_id=self.tokenizer.eos_token_id,
                stop_strings=[self.tokenizer.eos_token] + ["\n\n\n\n"],
                tokenizer=self.tokenizer,
            )
            formatted_answer = (
                self.tokenizer.decode(
                    out[0, inputs.shape[1] :], skip_special_tokens=True
                )
                .split("API Calls:", 1)[-1]
                .strip()
            )

        self.history.append(
            {
                "prompt": prompt,
                "response": {"answer": formatted_answer},
            }
        )

        return {formatted_answer}

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        return self.basic_request(prompt, **kwargs)
