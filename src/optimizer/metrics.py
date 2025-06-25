from dspy.evaluate import Evaluate
from datasets import Dataset

from evaluation.evaluator import Evaluation
from evaluation.utils import get_api_data, postprocess_preds


def exact_match(example, pred, trace=None):

    preds = [
        "\n".join(
            postprocess_preds(
                pred.answer.split("API Calls:")[-1].split("---")[0].splitlines()
            )
        )
    ]

    targets, predictions, example_ids, lengths = get_api_data(
        data=Dataset.from_dict(
            {
                "api_calls": [example.answer.splitlines()],
                "example_id": [0],
                "user_utterance": [example.user_utterance],
            }
        ),
        predictions=preds,
        alt_names_path="../data/locations.json",
    )


    evals = Evaluation(
        ground_truth=targets,
        predictions=predictions,
        datum_ids=example_ids,
        preprocess=True,
    )
    aggregate_metrics, instance_wise_metrics = evals.get_metrics(prefix="slot_")
    exact_match_scores = instance_wise_metrics["slot_em_scores"]
    for i, l in enumerate(lengths):
        if l:
            exact_match_scores[i] = 0.0
    new_exact_match = sum(exact_match_scores) / len(exact_match_scores)
    aggregate_metrics["slot_em_score"] = new_exact_match
    instance_wise_metrics["slot_em_scores"] = exact_match_scores

    answer_em = aggregate_metrics["slot_em_score"]

    print(
        f"\n\n[METRIC]\nUser Profile:\n{example.user_profile}\n\nUser Query;\n{example.user_utterance}\n\nTarget:\n{example.answer}\n\nPrediction:\n{pred.answer}\n\nEM: {answer_em}"
    )

    return answer_em


def f1(example, pred, trace=None):

    preds = ["\n".join(postprocess_preds(pred.answer.splitlines()))]

    targets, predictions, example_ids, lengths = get_api_data(
        data=Dataset.from_dict(
            {
                "api_calls": [example.answer.splitlines()],
                "example_id": [0],
                "user_utterance": [example.user_utterance],
            }
        ),
        predictions=preds,
        alt_names_path="../data/locations.json",
    )
    evals = Evaluation(
        ground_truth=targets,
        predictions=predictions,
        datum_ids=example_ids,
        preprocess=True,
    )
    aggregate_metrics, _ = evals.get_metrics(prefix="slot_")

    answer_f1 = aggregate_metrics["slot_f1_score"]

    print(
        f"User Profile:\n{example.user_profile}\n\nUser Query;\n{example.user_utterance}\n\nTarget:\n{example.answer}\n\nPrediction:\n{pred.answer}\n\nF1: {answer_f1}"
    )

    return answer_f1
