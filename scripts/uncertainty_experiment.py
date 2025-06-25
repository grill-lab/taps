import sys

sys.path.append("../src")

from argparse import ArgumentParser
from experiments.uncertainty_exp import UncertaintyExperiment


def main(args):

    exp = UncertaintyExperiment(args=args)
    exp.run()


if __name__ == "__main__":
    parser = ArgumentParser()

    # model args
    parser.add_argument(
        "--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3"
    )
    parser.add_argument("--load_in_4bit", type=bool, default=False)
    parser.add_argument("--use_endpoint", type=bool, default=False)
    parser.add_argument("--use_gpt", type=bool, default=False)

    # data args
    parser.add_argument("--test_dataset", type=str, default="../data/test.jsonl")
    parser.add_argument("--train_dataset", type=str, default="../data/train.jsonl")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument(
        "--tag_file_test", type=str, default="../data/gpt4_tag_2_shot.csv"
    )

    # experiment setup
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_shots", type=int, default=1)

    parser.add_argument("--setup_fpath", type=str, default=None)
    parser.add_argument("--use_setup_prompt", type=bool, default=False)
    parser.add_argument("--use_setup_shots", type=bool, default=False)
    parser.add_argument("--skip_first_demo", type=bool, default=False)
    parser.add_argument("--prompt", type=str, default="../prompts/default_prompt.txt")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--simple_aug", type=str, default=None)
    parser.add_argument("--inference_type", type=str, default="chat_completion")

    # uncertainty args
    parser.add_argument("--uncertainty_metric", type=str, default="least")
    parser.add_argument("--aug_preds_fpath", type=str)
    parser.add_argument("--scored_preds_fpath", type=str, default=None)

    # generation args
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=300)

    # output
    parser.add_argument("--output_dir", type=str, default="../../evaluation_results")

    args = parser.parse_args()

    main(args)
