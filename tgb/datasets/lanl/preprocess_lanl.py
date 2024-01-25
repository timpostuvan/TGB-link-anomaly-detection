"""
Code is adapted from: https://github.com/iHeartGraph/Euler/blob/main/lanl_experiments/loaders/split.py
"""

import os.path
from typing import Set, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

from tgb.utils.utils import save_pkl

# Please obtain the LANL data set from:
# https://csr.lanl.gov/data/cyber1/

ROOT_FOLDER = "/data/shares/stor02/tpostuvan/DATASETS/lanl/"

RED_TEAM_FILE = os.path.join(ROOT_FOLDER, "redteam.txt")
AUTH_FILE = os.path.join(ROOT_FOLDER, "auth.txt")
PROCESSED_FILE = os.path.join(ROOT_FOLDER, "ml_lanl.csv")
EDGE_FEAT_FILE = os.path.join(ROOT_FOLDER, "ml_lanl.npy")
SPLIT_FILE = os.path.join(ROOT_FOLDER, "split_ratios.txt")
VAL_ANOM_FILE = os.path.join(ROOT_FOLDER, "lanl_val_organic_anom_set_0.pkl")
TEST_ANOM_FILE = os.path.join(ROOT_FOLDER, "lanl_test_organic_anom_set_0.pkl")

comp2idx = dict()
next_idx = 0


def computer_to_index(comp: str) -> int:
    """
    Maps computer names to unique indices.

    Args:
        comp (str): The name of the computer.

    Returns:
        int: The unique index for the computer.
    """
    global next_idx
    if comp not in comp2idx:
        comp2idx[comp] = next_idx
        next_idx += 1

    return comp2idx[comp]


def save_anomalies_for_tgb(
    file_name: str,
    t: np.array,
    src: np.array,
    dst: np.array,
    msg: np.array = None,
):
    """
    Save anomalies in format for TGB datasets.

    Args:
        file_name (str): The name of the file to save the anomalies.
        t (np.array): Timestamps.
        src (np.array): Source indices.
        dst (np.array): Destination indices.
        msg (np.array, optional): Message data. Defaults to None.
    """
    evaluation_set = {}
    edge_label = np.zeros(src.shape[0])

    # Set weights and edge indices to dummy values.
    weights = np.ones(src.shape[0])
    edge_idxs = -1 * np.ones(src.shape[0])

    # If there are no messages, create a dummy vector.
    if msg is None:
        msg = np.zeros((src.shape[0], 1))

    evaluation_set = {
        "sources": src,
        "destinations": dst,
        "timestamps": t,
        "edge_idxs": edge_idxs,
        "edge_feat": msg,
        "w": weights,
        "edge_label": edge_label,
    }

    save_pkl(evaluation_set, file_name)


def save_benign_edges_for_tgb(
    file_name: str,
    edge_file_name: str,
    val_time: int,
    test_time: int,
    t: np.array,
    src: np.array,
    dst: np.array,
    msg: Optional[np.array] = None,
    num_samples: Optional[int] = 10**6,
):
    """
    Save benign edges in format for TGB datasets.

    Args:
        file_name (str): The name of the file to save the benign edges.
        edge_file_name (str): The name of the edge feature file.
        val_time (int): Validation time threshold.
        test_time (int): Test time threshold.
        t (np.array): Timestamps.
        src (np.array): Source indices.
        dst (np.array): Destination indices.
        msg (np.array, optional): Message data. Defaults to None.
        num_samples (int, optional): Number of samples. Defaults to 10^6.
    """
    # Equidistantly subsample the benign edges.
    step = max(src.shape[0] // num_samples, 1)
    t = t[::step]
    src = src[::step]
    dst = dst[::step]
    if msg is not None:
        msg = msg[::step, :]

    edge_label = np.zeros(src.shape[0])
    edge_idxs = np.arange(src.shape[0])

    # If there are no messages, create a dummy vector.
    if msg is None:
        msg = np.zeros((src.shape[0], 1))

    df = pd.DataFrame(
        data={
            "u": src,
            "i": dst,
            "ts": t,
            "label": edge_label,
            "idx": edge_idxs,
        }
    )
    df.to_csv(file_name, index=True)
    np.save(edge_file_name, arr=msg)

    # Determine proportions of validation and test sets.
    val_mask = np.logical_and(t <= test_time, t > val_time)
    test_mask = t > test_time

    val_ratio = np.sum(val_mask.astype(int)) / src.shape[0]
    test_ratio = np.sum(test_mask.astype(int)) / src.shape[0]
    with open(SPLIT_FILE, "w+") as f:
        f.write(f"Validation ratio: {val_ratio}\n")
        f.write(f"Test ratio: {test_ratio}\n")


def load_anomalies() -> Set:
    """
    Parses the redteam file and creates a set of anomalous edges.

    Returns:
        Set: Set of anomalies.
    """

    with open(RED_TEAM_FILE, "r") as f:
        red_events = f.read().split()

    anom_t, anom_src, anom_dst = [], [], []
    anom_set = set()
    for event in red_events:
        tokens = event.split(",")
        t, src, dst = (
            int(tokens[0]),
            computer_to_index(tokens[2]),
            computer_to_index(tokens[3]),
        )

        # Remove self-loops.
        if src != dst:
            anom_set.add((t, src, dst))
            anom_t.append(t)
            anom_src.append(src)
            anom_dst.append(dst)

    anom_t = np.array(anom_t)
    anom_src = np.array(anom_src)
    anom_dst = np.array(anom_dst)

    # Split anomalies into half for validation and testing sets.
    val_time, test_time = np.quantile(anom_t, q=[0.0, 0.5])
    val_mask = anom_t <= test_time
    test_mask = anom_t > test_time

    save_anomalies_for_tgb(
        file_name=VAL_ANOM_FILE,
        t=anom_t[val_mask],
        src=anom_src[val_mask],
        dst=anom_dst[val_mask],
    )

    save_anomalies_for_tgb(
        file_name=TEST_ANOM_FILE,
        t=anom_t[test_mask],
        src=anom_src[test_mask],
        dst=anom_dst[test_mask],
    )

    return anom_set, val_time, test_time


def preprocess():
    """
    Preprocesses LANL dataset in format for TGB datasets.
    """
    anom_set, val_time, test_time = load_anomalies()
    max_t = max([t for t, _, _ in anom_set])

    f = open(AUTH_FILE, "r")
    line = f.readline()

    prog = tqdm(desc="Seconds parsed", total=5011199)

    benign_t, benign_src, benign_dst = [], [], []

    last_time = 1
    while line:
        # Filter for better FPR/less Kerb noise.
        if "NTLM" not in line.upper():
            line = f.readline()
            continue

        tokens = line.split(",")
        t, src, dst = (
            int(tokens[0]),
            computer_to_index(tokens[3]),
            computer_to_index(tokens[4]),
        )

        # TODO: Potentially add some edge features.

        # Remove temporal edges that are anomalous, self-loops or occur after the last anomaly.
        if ((t, src, dst) not in anom_set) and (src != dst) and (t <= max_t):
            benign_t.append(t)
            benign_src.append(src)
            benign_dst.append(dst)

        if t != last_time:
            prog.update(t - last_time)
            last_time = t

        line = f.readline()

    f.close()

    benign_t = np.array(benign_t)
    benign_src = np.array(benign_src)
    benign_dst = np.array(benign_dst)

    save_benign_edges_for_tgb(
        file_name=PROCESSED_FILE,
        edge_file_name=EDGE_FEAT_FILE,
        val_time=val_time,
        test_time=test_time,
        t=benign_t,
        src=benign_src,
        dst=benign_dst,
        num_samples=10**6,
    )


if __name__ == "__main__":
    preprocess()
