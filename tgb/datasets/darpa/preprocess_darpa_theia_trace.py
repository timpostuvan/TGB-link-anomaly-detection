"""
Code for preprocessing DARPA-THEIA and DARPA-TRACE datasets from https://openreview.net/pdf?id=88tGIxxhsf.
"""

import os.path
from typing import Optional
import torch
import numpy as np
import pandas as pd

from tgb.utils.utils import save_pkl


class DatasetPaths:
    def __init__(self, root_folder, dataset_name):
        self.ROOT_FOLDER = root_folder
        self.INPUT_FILE = os.path.join(self.ROOT_FOLDER, "data.pt")
        self.PROCESSED_FILE = os.path.join(self.ROOT_FOLDER, f"ml_{dataset_name}.csv")
        self.EDGE_FEAT_FILE = os.path.join(self.ROOT_FOLDER, f"ml_{dataset_name}.npy")
        self.SPLIT_FILE = os.path.join(self.ROOT_FOLDER, "split_ratios.txt")
        self.VAL_ANOM_FILE = os.path.join(
            self.ROOT_FOLDER, f"{dataset_name}_val_organic_anom_set_0.pkl"
        )
        self.TEST_ANOM_FILE = os.path.join(
            self.ROOT_FOLDER, f"{dataset_name}_test_organic_anom_set_0.pkl"
        )


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
    split_file_name: str,
    val_time: int,
    test_time: int,
    t: np.array,
    src: np.array,
    dst: np.array,
    msg: Optional[np.array] = None,
):
    """
    Save benign edges in format for TGB datasets.

    Args:
        file_name (str): The name of the file to save the benign edges.
        edge_file_name (str): The name of the edge feature file.
        split_file_name (str): The name of the file to save the proportions of validation and test sets.
        val_time (int): Validation time threshold.
        test_time (int): Test time threshold.
        t (np.array): Timestamps.
        src (np.array): Source indices.
        dst (np.array): Destination indices.
        msg (np.array, optional): Message data. Defaults to None.
    """
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
    with open(split_file_name, "w+") as f:
        f.write(f"Validation ratio: {val_ratio}\n")
        f.write(f"Test ratio: {test_ratio}\n")


def preprocess(paths: DatasetPaths):
    """
    Preprocesses a DARPA dataset in format for TGB datasets.

    Args:
        paths (DatasetPaths): Paths to files of the dataset.
    """
    graph = torch.load(paths.INPUT_FILE)[0]

    print("Data read.")

    src, dst, msg, t = map(np.array, [graph.src, graph.dst, graph.msg, graph.t])
    node_features = graph.x.numpy()
    anom_mask = graph.malicious.numpy()

    print("Data converted to numpy arrays.")

    assert np.all(t[:-1] <= t[1:]), "Edges are not ordered according to timestamps!"

    # Move node features onto edges.
    msg = np.concatenate([msg, node_features[src], node_features[dst]], axis=-1)
    print(f"Shape of the features: {msg.shape}.")

    # The edges are split in the following way:
    # - Train set: Edges before the first anomaly.
    # - Validation/test sets: The rest of edges is split so that each set
    #   contains half of the anomalies.
    anom_t = t[anom_mask]
    anom_src = src[anom_mask]
    anom_dst = dst[anom_mask]
    anom_msg = msg[anom_mask]
    val_time, test_time = np.quantile(anom_t, q=[0.0, 0.5])
    val_mask = anom_t <= test_time
    test_mask = anom_t > test_time

    save_anomalies_for_tgb(
        file_name=paths.VAL_ANOM_FILE,
        t=anom_t[val_mask],
        src=anom_src[val_mask],
        dst=anom_dst[val_mask],
        msg=anom_msg[val_mask],
    )

    save_anomalies_for_tgb(
        file_name=paths.TEST_ANOM_FILE,
        t=anom_t[test_mask],
        src=anom_src[test_mask],
        dst=anom_dst[test_mask],
        msg=anom_msg[test_mask],
    )

    print("Anomalous edges saved.")

    benign_t = t[~anom_mask]
    benign_src = src[~anom_mask]
    benign_dst = dst[~anom_mask]
    benign_msg = msg[~anom_mask]
    save_benign_edges_for_tgb(
        file_name=paths.PROCESSED_FILE,
        edge_file_name=paths.EDGE_FEAT_FILE,
        split_file_name=paths.SPLIT_FILE,
        val_time=val_time,
        test_time=test_time,
        t=benign_t,
        src=benign_src,
        dst=benign_dst,
        msg=benign_msg,
    )
    print("Benign edges saved.\n")


if __name__ == "__main__":
    # DARPA-TRACE
    TRACE_PATHS = DatasetPaths(
        "/data/shares/stor02/tpostuvan/DATASETS/darpa_trace", "darpa-trace"
    )
    preprocess(paths=TRACE_PATHS)

    # DARPA-THEIA
    THEIA_PATHS = DatasetPaths(
        "/data/shares/stor02/tpostuvan/DATASETS/darpa_theia", "darpa-theia"
    )
    preprocess(paths=THEIA_PATHS)
