import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExperimentArguments:
    exp_name: str = field(
        metadata={"help": "name of the experiment for saving outputs"}
    )
    model_name: str = field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        metadata={"help": "model name or path"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "tokenizer name or path"},
    )
    shots: Optional[str] = field(
        default=None,
        metadata={"help": "number of demonstrations to use for few-shot evaluation"},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={
            "help": "whether to use quantization when loading the interpretation model"
        },
    )
    device: Optional[str] = field(
        default=None, metadata={"help": "device used for computation"}
    )
    strategy: str = field(
        default="direct", metadata={"help": "evaluation strategy to use"}
    )
    selection_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "selection method to use for choosing relevant standing instructions"
        },
    )
    selection_model_name: Optional[str] = field(
        default=None, metadata={"help": "model to use for selection"}
    )
    selection_prompt: Optional[str] = field(
        default="../prompts/standing_instructions.txt",
        metadata={"help": "system prompt for selection"},
    )
    preprocessing: bool = field(
        default=False, metadata={"help": "whether to preprocess data for selection"}
    )
    selection_add_special_tokens: bool = field(
        default=True,
        metadata={"help": "whether to add special tokens to selection inputs"},
    )
    selection_max_length: Optional[int] = field(
        default=None,
        metadata={"help": "maximium number of tokens for the selection model"},
    )
    selection_max_new_tokens: int = field(
        default=200,
        metadata={"help": "maximum number of tokens to generate for selection"},
    )
    selection_multipass: bool = field(
        default=False,
        metadata={"help": "whether to do a second selection pass"},
    )
    select_demonstrations: bool = field(
        default=False,
        metadata={
            "help": "whethere to use selection models to choose similar demonstrations"
        },
    )
    threshold: str = field(
        default="0.5", metadata={"help": "threshold for choosing relevant instructions"}
    )
    top_n: Optional[str] = field(
        default=None, metadata={"help": "number of relevant instructions to extract"}
    )
    add_special_tokens: bool = field(
        default=True,
        metadata={"help": "whether to use special tokens in prompts or not"},
    )
    add_indent: bool = field(
        default=True,
        metadata={
            "help": "whether to add indent when generating json-formatted prompts"
        },
    )
    debug: bool = field(
        default=False,
        metadata={
            "help": "whether to turn on the debug mode or not. During the debug mode some additional prints will be added."
        },
    )
    alt_names_path: str = field(
        default="../data/locations.json",
        metadata={
            "help": "path to a json file with alternative names for named entities"
        },
    )
    max_length: int = field(
        default=2048, metadata={"help": "maximum length of the generation output"}
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "maximum number of test examples to process. Used for debugging."
        },
    )
    output_dir: str = field(
        default="../results/", metadata={"help": "path to save the evaluation results"}
    )
    save_inter_res: bool = field(
        default=True, metadata={"help": "whether to save intermediate results or not"}
    )
    seed: int = field(default=42, metadata={"help": "seed to set random state"})

    def __post_init__(self):
        if self.shots is None:
            self.shots = "0 1 2 4 8"
        self.shots = list(map(int, self.shots.split()))
        if self.threshold:
            self.threshold = list(map(float, self.threshold.split()))
        if self.top_n:
            self.top_n = list(map(int, self.top_n.split()))


@dataclass
class DataArguments:
    test_dataset: str = field(
        default="../data/test.jsonl",
        metadata={"help": "path to the dataset"},
    )
    train_dataset: str = field(
        default="../data/train.jsonl",
        metadata={"help": "path to the train dataset used for"},
    )
    system_prompt: str = field(
        default="../prompts/simple_prompt_v1.txt",
        metadata={
            "help": "path to file with the dataset schema or string with dataset schema"
        },
    )
    augment: bool = field(
        default=False,
        metadata={"help": "whether to augment standing instructions or not"},
    )
    aug_fpath: str = field(
        default="../results/tagging/gpt-4-turbo__instr__1__val.csv",
        metadata={"help": "path to the file with data augmentation"},
    )
    train_aug_fpath: str = field(
        default="../data/train_instructions_golden.csv",
        metadata={
            "help": "path to the file with golden data augmentation fro demonstrations"
        },
    )


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
class TaggerArguments:
    """Arguments for the instruction tagger model."""

    strategy: str = field(
        default="instr", metadata={"help": "what to augment: instr or dial"}
    )
    shot: int = field(
        default=2, metadata={"help": "how many demonstrations to show the model"}
    )
    split: str = field(default="test", metadata={"help": "data split to augment"})
    model_name: str = field(
        default="gpt-4-turbo", metadata={"help": "OpenAI model for tagging"}
    )

    data_dir: str = field(
        default="../data/", metadata={"help": "path to the folder with the data"}
    )
    prompt_fpath: str = field(
        default="../prompts/tagger/tagger_model.txt",
        metadata={"help": "path to the prompt file"},
    )

    selection_model_name: str = field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        metadata={"help": "hf model for demonstration selection"},
    )
    targets_fpath: str = field(
        default="../data/val_instructions_golden.csv",
        metadata={"help": "path to the file with golden for evaluation"},
    )
    alt_names_path: str = field(
        default="../data/locations.json",
        metadata={
            "help": "path to a json file with alternative names for named entities"
        },
    )
    output_dir: str = field(
        default="../results/tagger/", metadata={"help": "path to save the results"}
    )


@dataclass
class TagCheckerArguments:
    """Arguments for the tag validation model."""

    fpath: str = field(metadata={"help": "path to the file to check"})

    model_name: str = field(
        default="gpt-4-turbo", metadata={"help": "OpenAI model for tagging"}
    )
    prompt_fpath: str = field(
        default="../prompts/tagger/tag_checker.txt",
        metadata={"help": "path to the prompt file"},
    )
    tagger_prompt_fpath: str = field(
        default="../prompts/tagger/tagger_model.txt",
        metadata={"help": "path to the prompt file"},
    )
    cot: bool = field(
        default=True, metadata={"help": "whether to use CoT prompting or not"}
    )
    evaluate: bool = field(
        default=False, metadata={"help": "whether to evaluate the checker output"}
    )
    targets_fpath: str = field(
        default="../data/val_instructions_golden.csv",
        metadata={"help": "path to the file with golden for evaluation"},
    )
    alt_names_path: str = field(
        default="../data/locations.json",
        metadata={
            "help": "path to a json file with alternative names for named entities"
        },
    )


@dataclass
class OptimizerParameters:
    """Parameters for DSPy optimizers."""

    model_name: str = field(
        default="mistralai/Mistral-7B-Instruct-v0.3",
    )
    test_fpath: str = field(default="../data/val.jsonl")
    train_fpath: str = field(default="../data/train.jsonl")
    prompt_file: str = field(default="../prompts/dspy_simple.txt")
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


@dataclass
class TagPerturbationArguments:
    """Arguments for tag perturbations experiments."""

    strategy: str = field(
        metadata={
            "help": "what to augment: 'action', 'slot_missing', 'semantic_slot', 'spurious_slot', or 'slot_boundary'"
        }
    )
    n_pct: int = field(default=10, metadata={"help": "% of eligible tags to corrupt"})
    seed: int = field(default=42, metadata={"help": "random seed"})
    input_fpath: str = field(
        default="../data/val_instructions_golden.csv",
        metadata={"help": "path to the file with golden instructions"},
    )
    api_schema_path: str = field(
        default="../data/api_schema.json",
        metadata={"help": "path to the file with the API schema"},
    )
    output_dir: str = field(
        default="../data/tagging/perturbations/",
        metadata={"help": "path to save the results"},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "maximum number of test examples to process. Used for debugging."
        },
    )