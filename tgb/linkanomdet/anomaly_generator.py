"""
Sample and generate anomalous edges that are going to be used for evaluation of a dynamic graph learning model.
Anomalies are generated and saved to files ONLY once for each type. Other times, they should be loaded from file.
"""
from typing import Optional, Dict, Any, Set, Tuple
from collections import defaultdict
import torch
import random
from torch import Tensor
import numpy as np
from torch_geometric.data import TemporalData
from tgb.utils.utils import (
    save_pkl,
    load_pkl,
    compute_cosine_distance,
    compute_l2_distance,
)
from tgb.utils.info import PROJ_DIR, DATA_VERSION_DICT, ANOMALY_ABBREVIATIONS
import os.path as osp
import os
import time
from tqdm import tqdm

from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader

from tgb.linkanomdet.dataset_pyg import PyGLinkAnomDetDataset


class AnomalousEdgeGenerator(object):
    def __init__(
        self,
        dataset_name: str,
        first_id: int,
        last_id: int,
        anom_prop: float,
        anom_type: str = "temporal-structural-contextual",
        anom_set_id: int = 0,
        rnd_seed: int = 123,
        historical_data: TemporalData = None,
    ) -> None:
        r"""
        Anomalous Edge Generator class
        This is a class for generating anomalous samples for a specific dataset.
        The set of the positive samples are provided, the anomalous samples are generated according
        to one of the types and are saved for consistent evaluation across different methods.
        It is assumed that the nodes are indexed sequentially with 'first_id'
        and 'last_id' being the first and last index, respectively.

        Parameters:
            dataset_name: Name of the dataset.
            first_id: Identity of the first node.
            last_id: Identity of the last node.
            anom_prop: Proportion of anomalous samples.
            anom_type: Type of anomalies to generate.
            anom_set_id: ID of anomaly set to generate (multiple sets of anomalies are supported).
            rnd_seed: Random seed for consistency.
            historical_data: Previous records of the positive edges.

        Returns:
            None
        """
        self.rnd_seed = rnd_seed + anom_set_id
        np.random.seed(self.rnd_seed)
        random.seed(self.rnd_seed)
        self.dataset_name = dataset_name

        self.first_id = first_id
        self.last_id = last_id
        self.anom_prop = anom_prop
        assert (
            anom_type in ANOMALY_ABBREVIATIONS
        ), f"The supported types of anomalies are: {list(ANOMALY_ABBREVIATIONS.keys())}!"
        self.anom_type = anom_type
        if self.anom_type == "combination":
            self.possible_anom_types = [
                anom_type
                for anom_type in ANOMALY_ABBREVIATIONS.keys()
                if anom_type != "combination" and anom_type != "organic"
            ]
        else:
            self.possible_anom_types = [self.anom_type]

        self.anom_set_id = anom_set_id
        self.historical_data = historical_data

    def _calculate_edge_probabilities(
        self,
        src: np.ndarray,
        dst: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate probabilites of edges with respect to their counts.
        More frequent edges have lower probability.

        Args:
            src (np.ndarray): Source nodes.
            dst (np.ndarray): Destination nodes.

        Returns:
            np.ndarray: Edge probabilities.
        """
        edge_counts = defaultdict(int)
        for u, v in zip(list(src), list(dst)):
            edge_counts[(u, v)] += 1

        # Weight function needs to have (1 / x) term in the beginning, to compensate
        # for the fact that an edge with x occurences, appears x times in the sampling set
        # (each time with a different message though).
        weight_func = lambda x: (1 / x) * (1 / np.sqrt(x))
        edge_weights = np.array(
            [weight_func(edge_counts[(u, v)]) for (u, v) in zip(list(src), list(dst))]
        )
        edge_p = edge_weights / edge_weights.sum()
        return edge_p

    def generate_anomalous_samples(
        self,
        data: TemporalData,
        split_mode: str,
        partial_path: str,
        **kwargs: Any,
    ) -> None:
        r"""
        Generate anomalous samples.

        Parameters:
            data: An object containing positive edges information.
            split_mode: Specifies whether to generate anomalous edges for 'validation' or 'test' split.
            partial_path: In which directory to save the generated anomalies.
        """
        version = 1
        if self.dataset_name in DATA_VERSION_DICT:
            version = DATA_VERSION_DICT[self.dataset_name]

        # File name for saving or loading
        filename = (
            partial_path
            + "/"
            + self.dataset_name
            + "_"
            + split_mode
            + "_"
            + ANOMALY_ABBREVIATIONS[self.anom_type]
            + "_anom"
            + f"_set_{self.anom_set_id}"
            + ("" if version == 1 else f"_v{version}")
            + ".pkl"
        )

        print(
            f"INFO: Type of edge anomaly: {self.anom_type}, ID of anomaly set: {self.anom_set_id}, Data Split: {split_mode}"
        )
        assert split_mode in [
            "val",
            "test",
        ], "Invalid split-mode! It should be `val` or `test`!"

        if os.path.exists(filename):
            print(
                f"INFO: {self.anom_type} anomalies with ID {self.anom_set_id} for '{split_mode}' evaluation are already generated!"
            )
        else:
            # Precompute various statistics that are needed for different anomaly injection strategies.
            print(
                f"INFO: Generating {self.anom_type} anomalies with ID {self.anom_set_id} for '{split_mode}' evaluation!"
            )
            # Retrieve the information from historical data.
            train_src, train_dst, train_msg, train_t = (
                self.historical_data.src.cpu().numpy(),
                self.historical_data.dst.cpu().numpy(),
                self.historical_data.msg.cpu().numpy(),
                self.historical_data.t.cpu().numpy(),
            )
            # Retrieve the information from evaluation data.
            eval_src, eval_dst, eval_msg, eval_t = (
                data.src.cpu().numpy(),
                data.dst.cpu().numpy(),
                data.msg.cpu().numpy(),
                data.t.cpu().numpy(),
            )

            num_anomalies = int(eval_src.shape[0] * self.anom_prop)

            # All possible sources and destinations, as well as time range
            all_src = np.arange(self.first_id, self.last_id + 1)
            all_dst = np.arange(self.first_id, self.last_id + 1)
            first_t, last_t = np.min(eval_t), np.max(eval_t)

            # All observed edges for checking structural consistency
            observed_edges = set(
                [
                    (src, dst)
                    for (src, dst) in zip(
                        list(train_src) + list(eval_src),
                        list(train_dst) + list(eval_dst),
                    )
                ]
            )

            # All observed edges with timestamps for checking temporal consistency
            observed_edges_with_t = set(
                [
                    (src, dst, t)
                    for (src, dst, t) in zip(
                        list(train_src) + list(eval_src),
                        list(train_dst) + list(eval_dst),
                        list(train_t) + list(eval_t),
                    )
                ]
            )

            eval_edge_p = self._calculate_edge_probabilities(eval_src, eval_dst)

            anomalies_src, anomalies_dst, anomalies_t, anomalies_msg = (
                [],
                [],
                [],
                [],
            )
            for _ in tqdm(range(num_anomalies), desc="Generating anomalies"):
                cur_anom_type = random.choice(self.possible_anom_types)

                if cur_anom_type == "temporal-structural-contextual":
                    (
                        anom_src,
                        anom_dst,
                        anom_msg,
                        anom_t,
                    ) = self.generate_temporal_structural_contextual_anomaly(
                        all_src=all_src,
                        all_dst=all_dst,
                        msg=eval_msg,
                        first_t=first_t,
                        last_t=last_t,
                        t=eval_t,
                        observed_edges=observed_edges,
                    )
                elif cur_anom_type == "temporal":
                    (
                        anom_src,
                        anom_dst,
                        anom_msg,
                        anom_t,
                    ) = self.generate_temporal_anomaly(
                        src=eval_src,
                        dst=eval_dst,
                        msg=eval_msg,
                        t=eval_t,
                        edge_p=eval_edge_p,
                        first_t=first_t,
                        last_t=last_t,
                        observed_edges_with_t=observed_edges_with_t,
                    )
                elif cur_anom_type == "structural-contextual":
                    (
                        anom_src,
                        anom_dst,
                        anom_msg,
                        anom_t,
                    ) = self.generate_structural_contextual_anomaly(
                        src=eval_src,
                        dst=eval_dst,
                        msg=eval_msg,
                        t=eval_t,
                        edge_p=eval_edge_p,
                        observed_edges=observed_edges,
                        **kwargs,
                    )
                elif cur_anom_type == "contextual":
                    (
                        anom_src,
                        anom_dst,
                        anom_msg,
                        anom_t,
                    ) = self.generate_contextual_anomaly(
                        src=eval_src,
                        dst=eval_dst,
                        msg=eval_msg,
                        t=eval_t,
                        edge_p=eval_edge_p,
                        **kwargs,
                    )
                elif cur_anom_type == "temporal-contextual":
                    (
                        anom_src,
                        anom_dst,
                        anom_msg,
                        anom_t,
                    ) = self.generate_temporal_contextual_anomaly(
                        src=eval_src,
                        dst=eval_dst,
                        msg=eval_msg,
                        t=eval_t,
                        edge_p=eval_edge_p,
                        first_t=first_t,
                        last_t=last_t,
                        **kwargs,
                    )
                else:
                    raise ValueError("Unsupported type of anomaly!")

                anomalies_src.append(anom_src)
                anomalies_dst.append(anom_dst)
                anomalies_msg.append(anom_msg)
                anomalies_t.append(anom_t)

            anomalies_src = np.array(anomalies_src)
            anomalies_dst = np.array(anomalies_dst)
            anomalies_msg = np.stack(anomalies_msg)
            anomalies_t = np.array(anomalies_t)

            evaluation_set = {}
            anomalies_edge_label = np.zeros(num_anomalies)

            # Set weights and edge indices to dummy values.
            anomalies_weights = np.ones(num_anomalies)
            anomalies_edge_idxs = -1 * np.ones(num_anomalies)

            evaluation_set = {
                "sources": anomalies_src,
                "destinations": anomalies_dst,
                "timestamps": anomalies_t,
                "edge_idxs": anomalies_edge_idxs,
                "edge_feat": anomalies_msg,
                "w": anomalies_weights,
                "edge_label": anomalies_edge_label,
            }

            # Save the generated evaluation set to disk.
            save_pkl(evaluation_set, filename)

    def generate_temporal_structural_contextual_anomaly(
        self,
        all_src: np.ndarray,
        all_dst: np.ndarray,
        msg: np.ndarray,
        first_t: int,
        last_t: int,
        t: np.ndarray,
        observed_edges: Set[Tuple[int, int]],
    ) -> Tuple[int, int, np.ndarray, float]:
        r"""
        Generate a temporal-structural-contextual edge anomaly based on the following strategy:
        - Randomly sample an edge and assign it a random in-distribution message
          and a random timestamp within the time range.
        - Check that the edge is a structural anomaly (i.e., it does not appear in observed edges).

        Args:
            all_src (np.ndarray): All possible source nodes.
            all_dst (np.ndarray): All possible destination nodes.
            msg (np.ndarray): Edge features.
            first_t (int): Lower bound for generation of the timestamp.
            last_t (int): Upper bound for generation of the timestamp.
            t (np.ndarray): Timestamps.
            observed_edges (Set[Tuple[int, int]]): All observed edges for checking structural consistency.

        Returns:
            Tuple[int, int, np.ndarray, float]: Generated temporal-structural-contextual anomaly.
        """
        # Sample a random node pair that hasn't been observed.
        # If no suitable node pair is found in 100 tries, just take the last one.
        anom_src, anom_dst, num_tries = None, None, 0
        while (anom_src is None) or (
            ((anom_src, anom_dst) in observed_edges) and (num_tries < 100)
        ):
            anom_src = np.random.choice(all_src, 1)[0]
            anom_dst = np.random.choice(all_dst, 1)[0]
            num_tries += 1

        # Sample a random time.
        anom_t = np.random.uniform(first_t, last_t, 1).astype(t.dtype)[0]

        # Sample a random message.
        msg_ind = np.random.choice(msg.shape[0], 1)[0]
        anom_msg = msg[msg_ind, :]
        return anom_src, anom_dst, anom_msg, anom_t

    def generate_temporal_anomaly(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        msg: np.ndarray,
        t: np.ndarray,
        edge_p: np.ndarray,
        first_t: int,
        last_t: int,
        observed_edges_with_t: Set[Tuple[int, int, float]],
    ) -> Tuple[int, int, np.ndarray, float]:
        r"""
        Generate a temporal edge anomaly based on the following strategy:
        - Randomly sample an edge that has appeared before, but assign it a random timestamp.
          For an easier task, an edge that appears more frequently has lower probability of being sampled.

        Args:
            src (np.ndarray): Source nodes.
            dst (np.ndarray): Destination nodes.
            msg (np.ndarray): Edge features.
            t (np.ndarray): Timestamps.
            edge_p (np.ndarray): Probability of sampling each of the edges.
            first_t (int): Lower bound for generation of the timestamp.
            last_t (int): Upper bound for generation of the timestamp.
            observed_edges_with_t (Set[Tuple[int, int, float]]): All observed edges with timestamps
                for checking temporal consistency.

        Returns:
            Tuple[int, int, np.ndarray, float]: Generated temporal anomaly.
        """
        # Sample a random edge.
        ind = np.random.choice(src.shape[0], 1, p=edge_p)[0]
        anom_src, anom_dst, anom_msg = src[ind], dst[ind], msg[ind]

        # Sample a random time and make sure that a benign edge doesn't appear at that time.
        # If no suitable time is found in 100 tries, just take the last one.
        anom_t, num_tries = None, 0
        while (anom_t is None) or (
            ((anom_src, anom_dst, anom_t) in observed_edges_with_t)
            and (num_tries < 100)
        ):
            anom_t = np.random.uniform(first_t, last_t, 1).astype(t.dtype)[0]
            num_tries += 1

        return anom_src, anom_dst, anom_msg, anom_t

    def _sample_contextually_inconsistent_attributes(
        self,
        cur_msg: np.ndarray,
        msg: np.ndarray,
        distance_metric: Optional[str] = "cosine",
        sample_size: Optional[int] = 10,
    ) -> np.ndarray:
        r"""
        Sample `sample_size` attributes and return the most dissimilar ones according to `distance_metric`.

        Args:
            cur_msg (np.ndarray): The current edge features.
            msg (np.ndarray): All edge features.
            distance_metric (Optional[str], optional): Distance metric for comparison of attributes. Defaults to "cosine".
            sample_size (Optional[int], optional): Number of attribute samples for comparison. Defaults to 10.

        Returns:
            np.ndarray: Contextually inconsistent attributes.
        """
        # Sample random edge attributes.
        mask = np.random.choice(msg.shape[0], sample_size, replace=False)
        selected_msg = msg[mask, :]

        distances = None
        if distance_metric == "cosine":
            distances = compute_cosine_distance(cur_msg, selected_msg)
        elif distance_metric == "l2":
            distances = compute_l2_distance(cur_msg, selected_msg)
        else:
            raise ValueError(f"{distance_metric} is not supported!")

        # Return attributes that are most dissimilar according to `distance_metric`.
        most_dissimilar_ind = np.argmax(distances)
        return selected_msg[most_dissimilar_ind]

    def generate_structural_contextual_anomaly(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        msg: np.ndarray,
        t: np.ndarray,
        edge_p: np.ndarray,
        observed_edges: Set[Tuple[int, int]],
        temporal_window_size: Optional[int] = 10,
        distance_metric: Optional[str] = "cosine",
        sample_size: Optional[int] = 10,
    ) -> Tuple[int, int, np.ndarray, float]:
        r"""
        Generate a structural-contextual edge anomaly based on the following strategy:
        - Randomly sample an anchor edge and a neighboring edge that is within a temporal window.
          For an easier task, an edge that appears more frequently has lower probability of being sampled.
        - Change destination node of the anchor edge with the destination node of the neighboring edge.
        - Check that the edge is a structural anomaly (i.e., it does not appear in observed edges).
        - Randomly sample attributes of `sample_size` other random edges and substitute attributes of the edge
          with the ones that are most dissimilar according to `distance_metric`.

        Args:
            src (np.ndarray): Source nodes.
            dst (np.ndarray): Destination nodes.
            msg (np.ndarray): Edge features.
            t (np.ndarray): Timestamps.
            edge_p (np.ndarray): Probability of sampling each of the edges.
            observed_edges (Set[Tuple[int, int]]): All observed edges for checking structural consistency.
            temporal_window_size (Optional[int], optional): Size of temporal window
                for sampling a neighboring edge. Defaults to 10.
            distance_metric (Optional[str], optional): Distance metric for comparison of attributes. Defaults to "cosine".
            sample_size (Optional[int], optional): Number of attribute samples for comparison. Defaults to 10.

        Returns:
            Tuple[int, int, np.ndarray, float]: Generated structural-contextual anomaly.
        """
        # Sample a random anchor edge.
        ind = np.random.choice(src.shape[0], 1, p=edge_p)[0]
        anom_src, cur_msg, anom_t = src[ind], msg[ind], t[ind]

        # Sample a random neighboring edge and make sure the new edge hasn't been observed.
        # If no suitable destination is found in 100 tries, just take the last one.
        anom_dst, num_tries = None, 0
        temporal_window = np.array(
            [
                x
                for x in range(-temporal_window_size, temporal_window_size + 1)
                if x != 0
            ]
        )
        while (anom_dst is None) or (
            ((anom_src, anom_dst) in observed_edges) and (num_tries < 100)
        ):
            neighbor_ind = np.clip(
                ind + np.random.choice(temporal_window, 1)[0],
                a_min=0,
                a_max=src.shape[0] - 1,
            )
            anom_dst = dst[neighbor_ind]
            num_tries += 1

        anom_msg = self._sample_contextually_inconsistent_attributes(
            cur_msg=cur_msg,
            msg=msg,
            distance_metric=distance_metric,
            sample_size=sample_size,
        )

        return anom_src, anom_dst, anom_msg, anom_t

    def generate_contextual_anomaly(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        msg: np.ndarray,
        t: np.ndarray,
        edge_p: np.ndarray,
        distance_metric: Optional[str] = "cosine",
        sample_size: Optional[int] = 10,
    ) -> Tuple[int, int, np.ndarray, float]:
        r"""
        Generate a contextual edge anomaly based on the following strategy:
        - Randomly sample an edge and attributes of `sample_size` other random edges.
          For an easier task, an edge that appears more frequently has lower probability of being sampled.
        - Substitute attributes of the edge with the ones that are most dissimilar according to `distance_metric`.

        Args:
            src (np.ndarray): Source nodes.
            dst (np.ndarray): Destination nodes.
            msg (np.ndarray): Edge features.
            t (np.ndarray): Timestamps.
            edge_p (np.ndarray): Probability of sampling each of the edges.
            distance_metric (Optional[str], optional): Distance metric for comparison of attributes. Defaults to "cosine".
            sample_size (Optional[int], optional): Number of attribute samples for comparison. Defaults to 10.

        Returns:
            Tuple[int, int, np.ndarray, float]: Generated contextual anomaly.
        """
        # Sample a random edge.
        ind = np.random.choice(src.shape[0], 1, p=edge_p)[0]
        anom_src, anom_dst, cur_msg, anom_t = src[ind], dst[ind], msg[ind], t[ind]

        anom_msg = self._sample_contextually_inconsistent_attributes(
            cur_msg=cur_msg,
            msg=msg,
            distance_metric=distance_metric,
            sample_size=sample_size,
        )

        return anom_src, anom_dst, anom_msg, anom_t

    def generate_temporal_contextual_anomaly(
        self,
        src: np.ndarray,
        dst: np.ndarray,
        msg: np.ndarray,
        t: np.ndarray,
        edge_p: np.ndarray,
        first_t: int,
        last_t: int,
        distance_metric: Optional[str] = "cosine",
        sample_size: Optional[int] = 10,
    ) -> Tuple[int, int, np.ndarray, float]:
        r"""
        Generate a temporal-contextual edge anomaly based on the following strategy:
        - Randomly sample an edge and attributes of `sample_size` other random edges.
          For an easier task, an edge that appears more frequently has lower probability of being sampled.
        - Substitute attributes of the edge with the ones that are most dissimilar according to `distance_metric`.
        - Assign the edge a random timestamp within the time range.

        Args:
            src (np.ndarray): Source nodes.
            dst (np.ndarray): Destination nodes.
            msg (np.ndarray): Edge features.
            t (np.ndarray): Timestamps.
            edge_p (np.ndarray): Probability of sampling each of the edges.
            first_t (int): Lower bound for generation of the timestamp.
            last_t (int): Upper bound for generation of the timestamp.
            distance_metric (Optional[str], optional): Distance metric for comparison of attributes. Defaults to "cosine".
            sample_size (Optional[int], optional): Number of attribute samples for comparison. Defaults to 10.

        Returns:
            Tuple[int, int, np.ndarray, float]: Generated temporal-contextual anomaly.
        """
        # Sample a random edge.
        ind = np.random.choice(src.shape[0], 1, p=edge_p)[0]
        anom_src, anom_dst, cur_msg = src[ind], dst[ind], msg[ind]

        anom_msg = self._sample_contextually_inconsistent_attributes(
            cur_msg=cur_msg,
            msg=msg,
            distance_metric=distance_metric,
            sample_size=sample_size,
        )

        # Sample a random time.
        anom_t = np.random.uniform(first_t, last_t, 1).astype(t.dtype)[0]

        return anom_src, anom_dst, anom_msg, anom_t
