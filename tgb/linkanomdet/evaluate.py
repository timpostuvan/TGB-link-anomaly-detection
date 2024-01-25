"""
Evaluator Module for Link Anomaly Detection.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tgb.utils.utils import recall_at_k

try:
    import torch
except ImportError:
    torch = None


class Evaluator(object):
    r"""Evaluator for Link Anomaly Detection."""

    def __init__(self, anom_label: int = 0):
        """
        Args:
            anom_label (int, optional): The label of anomaly class. Defaults to 0.
        """
        self.anom_label = anom_label
        self.valid_metric_list = ["auc", "ap", "recall@k"]

    def _parse_and_check_input(self, input_dict):
        r"""
        Check whether the input has the appropriate format.
        Parameters:
            input_dict: A dictionary containing "y_pred", "y_label", and "eval_metric".
            note: "eval_metric" should be a list including one or more of the followin metrics: ["auc", "ap", "recall@k"].
        Returns:
            y_pred: Predicted scores.
            y_label: Labels.
        """

        if "eval_metric" not in input_dict:
            raise RuntimeError("Missing key of eval_metric!")

        for eval_metric in input_dict["eval_metric"]:
            if eval_metric in self.valid_metric_list:
                if "y_pred" not in input_dict:
                    raise RuntimeError("Missing key of y_pred")
                if "y_label" not in input_dict:
                    raise RuntimeError("Missing key of y_label")

                y_pred, y_label = input_dict["y_pred"], input_dict["y_label"]

                # Converting to numpy on cpu.
                if torch is not None and isinstance(y_pred, torch.Tensor):
                    y_pred = y_pred.detach().cpu().numpy()
                if torch is not None and isinstance(y_label, torch.Tensor):
                    y_label = y_label.detach().cpu().numpy()

                # Check type and shape.
                if not isinstance(y_pred, np.ndarray) or not isinstance(
                    y_label, np.ndarray
                ):
                    raise RuntimeError(
                        "Arguments to Evaluator need to be either numpy ndarray or torch tensor!"
                    )
            else:
                print(
                    "ERROR: The evaluation metric should be in:", self.valid_metric_list
                )
                raise ValueError("Unsupported eval metric %s " % (eval_metric))
        self.eval_metric = input_dict["eval_metric"]

        return y_pred, y_label

    def _eval_auc_and_ap_and_recall_at_k(self, y_pred, y_label):
        r"""
        Compute AUC, AP and Recall@k metrics.

        Parameters:
            y_pred: Predicted scores.
            y_label: Labels.

        Returns:
            A dictionary containing the computed performance metrics.
        """

        auc_score = roc_auc_score(y_score=y_pred, y_true=y_label)

        # AP and Recall@k functions expect probability of positive class (i.e., anomalies).
        # However, predictions are generated based on link prediction so anomalous edges have low
        # scores instead of high ones. Therefore, predictions have to flipped.
        if self.anom_label == 0:
            y_pred = 1 - y_pred

        ap_score = average_precision_score(
            y_score=y_pred, y_true=y_label, pos_label=self.anom_label
        )
        recall_at_k_score = recall_at_k(
            y_score=y_pred, y_true=y_label, pos_label=self.anom_label
        )

        return {
            "auc": auc_score,
            "ap": ap_score,
            "recall@k": recall_at_k_score,
        }

    def eval(self, input_dict: dict) -> dict:
        r"""
        Evaluate on the anomaly detection task.
        This method is callable through an instance of this object to compute the metric.

        Parameters:
            input_dict: A dictionary containing "y_pred", "y_label", and "eval_metric".
                        The performance metric is calculated for the provided scores.
        Returns:
            perf_dict: A dictionary containing the computed performance metric.
        """
        assert all(
            [metric in self.valid_metric_list for metric in input_dict["eval_metric"]]
        ), f"Not all metrics in {input_dict['eval_metric']} are valid!"

        y_pred, y_label = self._parse_and_check_input(
            input_dict
        )  # Convert the predictions to numpy.
        all_metrics_dict = self._eval_auc_and_ap_and_recall_at_k(y_pred, y_label)
        perf_dict = {
            metric: all_metrics_dict[metric] for metric in input_dict["eval_metric"]
        }

        return perf_dict


if __name__ == "__main__":
    evaluator = Evaluator(anom_label=0)
    input_dict = {
        "y_pred": np.array([0, 0.5, 0, 1, 1]),
        "y_label": np.array([0, 0, 1, 1, 1]),
        "eval_metric": ["auc", "ap", "recall@k"],
    }
    print(evaluator.eval(input_dict))
