import os
import random
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model_and_tokenizer(model_name: str, load_in_4bit: bool = False):
    if not load_in_4bit:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    else:
        config = BitsAndBytesConfig(
            load_in_4bit=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=config, device_map="auto"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
