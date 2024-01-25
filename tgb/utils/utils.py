import numpy as np
import random
import os
import pickle
from typing import Any
import sys
import argparse
import json
import io


# import torch
def save_pkl(obj: Any, fname: str) -> None:
    r"""
    save a python object as a pickle file
    """
    with open(fname, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(fname: str) -> Any:
    r"""
    load a python object from a pickle file
    """
    with open(fname, "rb") as handle:
        return pickle.load(handle)


def set_random_seed(seed: int):
    r"""
    setting random seed for reproducibility
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def compute_cosine_distance(x: np.ndarray, y: np.ndarray):
    x_norm = x / np.linalg.norm(x, axis=-1, keepdims=True)
    y_norm = y / np.linalg.norm(y, axis=-1, keepdims=True)

    distances = (1 - np.dot(y_norm, x_norm)) / 2
    return distances


def compute_l2_distance(x: np.ndarray, y: np.ndarray):
    x = np.repeat(x.reshape(1, -1), y.shape[0], axis=0)
    distances = np.linalg.norm(x - y, axis=-1)
    return distances


def recall_at_k(y_score: np.ndarray, y_true: np.ndarray, pos_label: int = 1):
    """
    Compute Recall@k metric with k as the number of positive samples.

    Args:
        y_score (np.ndarray): Target scores for positive class.
        y_true (np.ndarray): True binary labels or binary label indicators.
        pos_label (int, optional): The label of the positive class. Defaults to 1.

    Returns:
        float: Recall@k score.
    """
    k = np.sum((y_true == pos_label).astype(int))

    # Take only the highest scores because `y_pred` contains scores for positive class.
    top_k_indices = np.argsort(y_score)[-k:]

    recall_at_k_score = np.sum((y_true[top_k_indices] == pos_label).astype(int)) / k
    return recall_at_k_score


def get_args():
    parser = argparse.ArgumentParser("*** TGB ***")
    parser.add_argument(
        "-d", "--data", type=str, help="Dataset name", default="tgbl-wiki"
    )
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-4)
    parser.add_argument("--bs", type=int, help="Batch size", default=200)
    parser.add_argument(
        "--k_value", type=int, help="k_value for computing ranking metrics", default=10
    )
    parser.add_argument("--num_epoch", type=int, help="Number of epochs", default=50)
    parser.add_argument("--seed", type=int, help="Random seed", default=1)
    parser.add_argument("--mem_dim", type=int, help="Memory dimension", default=100)
    parser.add_argument("--time_dim", type=int, help="Time dimension", default=100)
    parser.add_argument("--emb_dim", type=int, help="Embedding dimension", default=100)
    parser.add_argument(
        "--tolerance", type=float, help="Early stopper tolerance", default=1e-6
    )
    parser.add_argument(
        "--patience", type=float, help="Early stopper patience", default=5
    )
    parser.add_argument(
        "--num_run", type=int, help="Number of iteration runs", default=1
    )
    parser.add_argument(
        "--anom_type", type=str, help="Type of anomalies to load", default=None
    )

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv


def save_results(new_results: dict, filename: str):
    r"""
    save (new) results into a json file
    :param: new_results (dictionary): a dictionary of new results to be saved
    :filename: the name of the file to save the (new) results
    """
    if os.path.isfile(filename):
        # append to the file
        with open(filename, "r+") as json_file:
            file_data = json.load(json_file)
            # convert file_data to list if not
            if type(file_data) is dict:
                file_data = [file_data]
            file_data.append(new_results)
            json_file.seek(0)
            json.dump(file_data, json_file, indent=4)
    else:
        # dump the results
        with open(filename, "w") as json_file:
            json.dump(new_results, json_file, indent=4)
