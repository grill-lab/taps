import sys
import os

sys.path.append("../src")

from transformers import HfArgumentParser

from optimizer.pipeline import DSPyPipeline
from utils.arguments import GenerationParameters, OptimizerParameters


def main(args, gen_params):

    print("[ARGUMENTS] =", args)
    print(gen_params)

    exp = DSPyPipeline(
        model_name=args.model_name,
        gen_params=gen_params,
        prompt_file=args.prompt_file,
        tag=args.tag,
        api_in_prompt=args.api_in_prompt,
        metric=args.metric,
        num_threads=args.num_threads,
        max_bootstrapped_demos=args.max_bootstrapped_demos,
        max_labeled_demos=args.max_labeled_demos,
        num_candidate_programs=args.num_candidate_programs,
        output_dir=args.output_dir,
        load_in_4bit=args.load_in_4bit,
    )

    exp.run(
        test_fpath=args.test_fpath,
        train_fpath=args.train_fpath,
        train_profile_tags=args.train_profile_tags,
        test_profile_tags=args.test_profile_tags,
        max_samples=args.max_samples,
        max_samples_eval=args.max_samples_eval,
    )


if __name__ == "__main__":
    parser = HfArgumentParser([GenerationParameters, OptimizerParameters])

    gen_params, exp_args = parser.parse_args_into_dataclasses()

    main(exp_args, gen_params)
