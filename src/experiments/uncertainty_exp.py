import os
import pandas as pd
import numpy as np

from scipy.stats import pearsonr
from experiments.base_exp import BaseExperiment


class UncertaintyExperiment(BaseExperiment):
    def __init__(self, args):
        super().__init__(args=args)

    def get_output_fpath(self):

        model_name = self.args.model_name.replace("/", "_")
        test_split = self.args.test_dataset.split("/")[-1].split(".")[0]
        aug = (
            self.args.tag
            if self.args.tag is not None
            else f"simple_{self.args.simple_aug}" if self.args.simple_aug else "org"
        )

        res_dir = f"{self.args.output_dir}/{test_split}/{model_name}/uncertainty/"
        os.makedirs(res_dir, exist_ok=True)

        fname = (
            model_name
            + ("__endpoint__" if self.args.use_endpoint else "__")
            + self.args.inference_type
            + f"__{aug}"
            + "__"
            + self.args.prompt.split("/")[-1].replace(".txt", "")
            + "__"
            + f"{self.args.n_shots}_shots__"
            + self.args.suffix
            + ("__" if self.args.suffix else "")
            + self.args.test_dataset.split("/")[-1].split(".")[0]
        )

        previous_exps = list(filter(lambda x: x.startswith(fname), os.listdir(res_dir)))
        ind = len(previous_exps)

        fpath = os.path.join(res_dir, fname + f"_{ind}.json")
        return fpath

    @staticmethod
    def calculate_probability(sequence_score: np.array) -> float:
        return np.exp(sequence_score)

    @staticmethod
    def calculate_least_uncertainty(sequence_score: np.array) -> float:
        return 1 - UncertaintyExperiment.calculate_probability(sequence_score)

    @staticmethod
    def calculate_correlation(
        metrics: list, uncertainty: list, corr_func: callable
    ) -> tuple:
        corr = corr_func(uncertainty, metrics)
        return corr.statistic, corr.pvalue

    def get_scores(self) -> pd.DataFrame:
        model = self.load_model()
        test, train, prompt, test_instructions, train_instructions = self.prepare_data()

        predictions = model.get_model_predictions(
            test=test,
            train=train,
            prompt=prompt,
            tag=self.args.tag,
            simple_aug=self.args.simple_aug,
            dspy=self.args.setup_fpath is not None,
            test_instructions=test_instructions,
            train_instructions=train_instructions,
            n_shots=self.args.n_shots,
            max_tokens=self.args.max_tokens,
            do_sample=self.args.do_sample,
            num_beams=self.args.num_beams,
            temperature=self.args.temperature,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            seed=self.args.seed,
            inference_type=self.args.inference_type,
            stop_strings=["\n\n\n\n", "---"],
            max_samples=self.args.max_samples,
            output_scores=True,
        )

        # save experiment arguments
        predictions["args"] = [self.args.__dict__] * predictions.shape[0]

        # get metrics
        agg_metrics, instance_metrics = self.evaluate(
            predictions=predictions["reference_api_calls"].tolist(), test=test
        )
        predictions["em_score"] = instance_metrics["em_scores"]
        predictions["f1_score"] = instance_metrics["f1_scores"]
        predictions["prec_score"] = instance_metrics["prec_scores"]
        predictions["rec_score"] = instance_metrics["rec_scores"]
        predictions["avg_score"] = np.mean(
            [
                instance_metrics["em_scores"],
                instance_metrics["f1_scores"],
                instance_metrics["prec_scores"],
                instance_metrics["rec_scores"],
            ],
            axis=0,
        )

        agg_metrics["avg_score"] = np.mean(
            [
                agg_metrics["em_score"],
                agg_metrics["f1_score"],
                agg_metrics["prec_score"],
                agg_metrics["rec_score"],
            ]
        )

        uncertainty = predictions["score"].apply(self.calculate_least_uncertainty)
        predictions["uncertainty_score"] = uncertainty

        corr_val, p_val = self.calculate_correlation(
            metrics=predictions["f1_score"].tolist(),
            uncertainty=uncertainty.tolist(),
            corr_func=pearsonr,
        )
        agg_metrics["uncertainty_corr"] = {
            "statistic": corr_val,
            "p_val": p_val,
        }

        predictions["agg_metrics"] = [agg_metrics] * predictions.shape[0]

        # save evaluation results
        fpath = self.get_output_fpath()
        self.fpath = fpath
        print("\nSaving to file", fpath)

        predictions.to_json(
            fpath,
            orient="records",
            indent=4,
            force_ascii=False,
        )
        return predictions

    @staticmethod
    def calc_scores_by_t(
        predictions: pd.DataFrame, t: float, better_high: bool = False
    ) -> pd.DataFrame:
        for metric in ["em", "f1", "prec", "rec", "avg"]:
            if better_high:
                predictions[f"{metric}_score"] = predictions.apply(
                    lambda x: (
                        x[f"{metric}_score_aug"]
                        if x["uncertainty_score"] < t
                        else x[f"{metric}_score_org"]
                    ),
                    axis=1,
                )
            else:
                predictions[f"{metric}_score"] = predictions.apply(
                    lambda x: (
                        x[f"{metric}_score_aug"]
                        if x["uncertainty_score"] >= t
                        else x[f"{metric}_score_org"]
                    ),
                    axis=1,
                )

        if better_high:
            predictions["augmented"] = predictions["uncertainty_score"] < t
        else:
            predictions["augmented"] = predictions["uncertainty_score"] >= t

        agg_metrics = {
            "em_score": predictions["em_score"].mean(),
            "f1_score": predictions["f1_score"].mean(),
            "prec_score": predictions["prec_score"].mean(),
            "rec_score": predictions["rec_score"].mean(),
            "avg_score": predictions["avg_score"].mean(),
            "augmented": predictions["augmented"].sum(),
        }
        return agg_metrics

    @staticmethod
    def merge_preds(preds_1: pd.DataFrame, preds_2: pd.DataFrame) -> pd.DataFrame:
        predictions = preds_1.copy().merge(
            preds_2.copy(),
            on="example_id",
            suffixes=("_org", "_aug"),
        )
        return predictions

    @staticmethod
    def uncertainty_aware_sampling(
        default_predictions: pd.DataFrame,
        aug_predictions: pd.DataFrame,
        t: float,
        better_high: bool = False,
    ) -> pd.DataFrame:

        predictions = UncertaintyExperiment.merge_preds(
            preds_1=default_predictions, preds_2=aug_predictions
        )

        agg_metrics = UncertaintyExperiment.calc_scores_by_t(
            predictions=predictions, t=t, better_high=better_high
        )

        predictions["agg_metrics"] = [agg_metrics] * predictions.shape[0]

        return predictions

    @staticmethod
    def find_threshold(
        predictions, metric: str, better_high: bool = False, percent: bool = False
    ) -> float:
        best_t = 0
        best_score = 0
        if percent:
            threshold_space = range(0, 100, 1)
        else:
            threshold_space = np.linspace(0, 1, 100)
        for t in threshold_space:
            agg_metrics = UncertaintyExperiment.calc_scores_by_t(
                predictions=predictions, t=t, better_high=better_high
            )
            if agg_metrics[f"{metric}_score"] > best_score:
                best_score = agg_metrics[f"{metric}_score"]
                best_t = t

        return best_t

    @staticmethod
    def oracle_baseline(predictions: pd.DataFrame, metric: str) -> pd.DataFrame:

        better_scores_aug = predictions[
            predictions[f"{metric}_score_aug"] > predictions[f"{metric}_score_org"]
        ]
        better_scores_org = predictions[
            predictions[f"{metric}_score_org"] >= predictions[f"{metric}_score_aug"]
        ]

        scores = {}
        for metric in ["em", "f1", "prec", "rec"]:
            scores[f"{metric}_score"] = (
                better_scores_aug[f"{metric}_score_aug"].sum()
                + better_scores_org[f"{metric}_score_org"].sum()
            ) / len(predictions)
        scores["augmented"] = len(better_scores_aug)
        return scores

    def run(self):
        if self.args.scored_preds_fpath is None:
            unc_predictions = self.get_scores()
        else:
            unc_predictions = pd.read_json(self.args.scored_preds_fpath)
            self.fpath = self.args.scored_preds_fpath

        aug_predictions = pd.read_json(self.args.aug_preds_fpath)
        if "avg_score" not in aug_predictions["agg_metrics"][0]:
            agg_metrics = aug_predictions["agg_metrics"][0]
            agg_metrics["avg_score"] = np.mean(
                [
                    agg_metrics["em_score"],
                    agg_metrics["f1_score"],
                    agg_metrics["prec_score"],
                    agg_metrics["rec_score"],
                ]
            )
            aug_predictions["agg_metrics"] = [agg_metrics] * len(aug_predictions)

        predictions = UncertaintyExperiment.merge_preds(
            preds_1=unc_predictions, preds_2=aug_predictions
        )

        threshold = UncertaintyExperiment.find_threshold(
            predictions=predictions,
            metric="f1",
            better_high=self.args.uncertainty_metric != "least",
        )

        final_predictions = UncertaintyExperiment.uncertainty_aware_sampling(
            default_predictions=unc_predictions,
            aug_predictions=aug_predictions,
            t=threshold,
            better_high=self.args.uncertainty_metric != "least",
        )

        agg_scores = final_predictions["agg_metrics"].iloc[0]
        oracle_baseline = UncertaintyExperiment.oracle_baseline(
            predictions=final_predictions, metric="f1"
        )
        agg_scores["oracle"] = oracle_baseline
        final_predictions["agg_metrics"] = [agg_scores] * final_predictions.shape[0]

        print("\nSaving to file", self.fpath)
        final_predictions.to_json(
            self.fpath,
            orient="records",
            indent=4,
            force_ascii=False,
        )

        return final_predictions
