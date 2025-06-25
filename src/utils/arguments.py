import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerationParameters:
    """Parameters for generation."""

    max_new_tokens: int = field(
        default=100, metadata={"help": "maximum number of generated tokens"}
    )
    do_sample: bool = field(
        default=False, metadata={"help": "generation params: do_sample"}
    )
    early_stopping: bool = field(
        default=False, metadata={"help": "generation params: early_stopping"}
    )
    num_beams: int = field(default=1, metadata={"help": "generation params: num_beams"})
    temperature: float = field(
        default=1.0, metadata={"help": "generation params: temperature"}
    )
    top_k: int = field(default=50, metadata={"help": "generation params: top_k"})
    top_p: float = field(default=1.0, metadata={"help": "generation params: top_p"})
    typical_p: float = field(
        default=1.0, metadata={"help": "params for typical decoding"}
    )
    repetition_penalty: float = field(
        default=1.0, metadata={"help": "generation params: repetition_penalty"}
    )
    no_repeat_ngram_size: float = field(
        default=0, metadata={"help": "generation params: no_repeat_ngram_size"}
    )
    max_tokens: int = field(default=8192)


@dataclass
class OptimizerParameters:
    """Parameters for DSPy optimizers."""

    model_name: str = field(
        default="mistralai/Mistral-7B-Instruct-v0.3",
    )
    test_fpath: str = field(default="../data/val.jsonl")
    train_fpath: str = field(default="../data/train.jsonl")
    prompt_file: str = field(default="../prompts/default_prompt.txt")
    tag: Optional[str] = field(default=None)
    api_in_prompt: bool = field(default=True)
    train_profile_tags: str = field(default="../data/train_instructions_golden.csv")
    test_profile_tags: str = field(default="../data/gpt4_tag_2_shot.csv")
    metric: str = field(default="em")
    num_threads: int = field(default=1)
    max_bootstrapped_demos: int = field(default=5)
    max_labeled_demos: int = field(default=5)
    num_candidate_programs: int = field(default=10)
    max_samples: int = field(default=-1)
    max_samples_eval: int = field(default=-1)
    output_dir: str = field(default="../results/dspy/")
    load_in_4bit: bool = field(default=False)
