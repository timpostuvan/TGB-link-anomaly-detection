from typing import Optional, Dict, Any, Tuple, List
import os
import os.path as osp
import shutil
import numpy as np
import pandas as pd
import zipfile
import requests
from clint.textui import progress

from tgb.utils.info import (
    PROJ_DIR,
    DATA_URL_DICT,
    DATA_VERSION_DICT,
    ANOMALY_ABBREVIATIONS,
    DATA_ORGANIC_ANOMALIES,
    BColors,
)
from tgb.utils.pre_process import (
    csv_to_pd_data,
    process_node_feat,
    csv_to_pd_data_sc,
    csv_to_pd_data_rc,
    load_edgelist_wiki,
    load_preprocessed_data,
)
from tgb.utils.utils import save_pkl, load_pkl
from tgb.utils.graph_generation import generate_synthetic_graph


class LinkAnomDetDataset(object):
    def __init__(
        self,
        name: str,
        root: Optional[str] = "datasets",
        absolute_path: Optional[bool] = False,
        meta_dict: Optional[dict] = None,
        preprocess: Optional[bool] = True,
        val_ratio: Optional[float] = 0.15,
        test_ratio: Optional[float] = 0.15,
        **kwargs,
    ):
        r"""Dataset class for link anomaly detection. Stores meta information about each dataset such as evaluation metrics etc.
        It also automatically pre-processes the dataset.
        Args:
            name: Name of the dataset.
            root: Root directory to store the dataset folder.
            meta_dict: Dictionary containing meta information about the dataset, should contain key 'dir_name' which is the name of the dataset folder.
            preprocess: Whether to pre-process the dataset.
            val_ratio: Ratio of validation data.
            test_ratio: Ratio of test data.
        """
        self.name = name  # Original name
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Check if dataset url exists.
        if self.name in DATA_URL_DICT:
            self.url = DATA_URL_DICT[self.name]
        else:
            self.url = None
            print(
                f"Dataset {self.name} url not found. The dataset has to be either downloaded manually or generated."
            )

        self.metric = ["auc", "ap", "recall@k"]

        if not absolute_path:
            root = PROJ_DIR + root

        if meta_dict is None:
            self.dir_name = "_".join(name.split("-"))  # Replace hyphen with underline.
            meta_dict = {"dir_name": self.dir_name}
        else:
            self.dir_name = meta_dict["dir_name"]
        self.root = osp.join(root, self.dir_name)
        self.meta_dict = meta_dict

        # Whether the dataset is not part of the original TGB dataset.
        self.custom_dataset = False if (self.name[:3] == "tgb") else True

        if self.custom_dataset:
            if "fname" not in self.meta_dict:
                self.meta_dict["fname"] = f"{self.root}/ml_{self.name}.csv"
                self.meta_dict["fname_edge"] = f"{self.root}/ml_{self.name}.npy"
                self.meta_dict["nodefile"] = None
        else:
            if "fname" not in self.meta_dict:
                self.meta_dict["fname"] = self.root + "/" + self.name + "_edgelist.csv"
                self.meta_dict["nodefile"] = None

            if name == "tgbl-flight":
                self.meta_dict["nodefile"] = self.root + "/" + "airport_node_feat.csv"

        self.meta_dict["val_anomalies"] = (
            self.root + "/" + self.name + "_val_{anom_type}_anom_set_{anom_set_id}.pkl"
        )
        self.meta_dict["test_anomalies"] = (
            self.root + "/" + self.name + "_test_{anom_type}_anom_set_{anom_set_id}.pkl"
        )

        # Version check
        self.version_passed = True
        self._version_check()

        # Initialization
        self._node_feat = None
        self._edge_feat = None
        self._full_data = None
        self._train_data = None
        self._val_data = None
        self._test_data = None
        self._anomalies_loaded = {"val": None, "test": None}

        if self.url is not None:
            self.download()
        elif self.name == "synthetic":
            self.obtain_synthetic_dataset(**kwargs)
        else:
            if not osp.exists(self.meta_dict["fname"]):
                raise FileNotFoundError(
                    f"The dataset has to be downloaded manually, but it is not found at {self.meta_dict['fname']}"
                )

        # Check if the root directory exists, if not create it.
        if osp.isdir(self.root):
            print("Dataset directory is ", self.root)
        else:
            # os.makedirs(self.root)
            raise FileNotFoundError(f"Directory not found at {self.root}")

        if preprocess:
            self.pre_process()

    def _version_check(self) -> None:
        r"""Implement version checks for dataset files.
        Updates the file names based on the current version number.
        Prompt the user to download the new version via self.version_passed variable.
        """
        if self.name in DATA_VERSION_DICT:
            version = DATA_VERSION_DICT[self.name]
        else:
            print(f"Dataset {self.name} version number not found.")
            self.version_passed = False
            return None

        if version > 1:
            # Check if current version is outdated.
            self.meta_dict["fname"] = (
                self.root + "/" + self.name + "_edgelist_v" + str(int(version)) + ".csv"
            )
            self.meta_dict["nodefile"] = None
            if self.name == "tgbl-flight":
                self.meta_dict["nodefile"] = (
                    self.root + "/" + "airport_node_feat_v" + str(int(version)) + ".csv"
                )

            self.meta_dict["val_anomalies"] = (
                self.root
                + "/"
                + self.name
                + "_val_{anom_type}_anom"
                + "_set_{anom_set_id}"
                + "_v"
                + str(int(version))
                + ".pkl"
            )
            self.meta_dict["test_anomalies"] = (
                self.root
                + "/"
                + self.name
                + "_test_{anom_type}_anom"
                + "_set_{anom_set_id}"
                + "_v"
                + str(int(version))
                + ".pkl"
            )

            if not osp.exists(self.meta_dict["fname"]):
                print(f"Dataset {self.name} version {int(version)} not found.")
                print(f"Please download the latest version of the dataset.")
                self.version_passed = False
                return None

    def download(self):
        """
        Downloads this dataset from url.
        Check if files are already downloaded.
        """
        # Check if the file already exists.
        if osp.exists(self.meta_dict["fname"]):
            print("Raw file found, skipping download.")
            return

        inp = input(
            "Will you download the dataset(s) now? (y/N)\n"
        ).lower()  # Ask if the user wants to download the dataset.

        if inp == "y":
            print(
                f"{BColors.WARNING}Download started, this might take a while . . . {BColors.ENDC}"
            )
            print(f"Dataset title: {self.name}")

            if self.url is None:
                raise Exception("Dataset url not found, download not supported yet.")
            else:
                r = requests.get(self.url, stream=True)
                # download_dir = self.root + "/" + "download"
                if osp.isdir(self.root):
                    print("Dataset directory is ", self.root)
                else:
                    os.makedirs(self.root)

                path_download = self.root + "/" + self.name + ".zip"
                with open(path_download, "wb") as f:
                    total_length = int(r.headers.get("content-length"))
                    for chunk in progress.bar(
                        r.iter_content(chunk_size=1024),
                        expected_size=(total_length / 1024) + 1,
                    ):
                        if chunk:
                            f.write(chunk)
                            f.flush()

                # TODO: This doesn't work for Reddit dataset for some reason.
                # For unzipping the file
                with zipfile.ZipFile(path_download, "r") as zip_ref:
                    zip_ref.extractall(self.root)

                # Move extracted files of a custom datasets into the dedicated folder.
                if self.custom_dataset:
                    cur_df = f"{self.root}/{self.name}/ml_{self.name}.csv"
                    cur_edge_features = f"{self.root}/{self.name}/ml_{self.name}.npy"

                    os.rename(src=cur_df, dst=self.meta_dict["fname"])
                    os.rename(src=cur_edge_features, dst=self.meta_dict["fname_edge"])
                    shutil.rmtree(f"{self.root}/{self.name}")

                print(f"{BColors.OKGREEN}Download completed {BColors.ENDC}")
                self.version_passed = True
        else:
            raise Exception(
                BColors.FAIL + "Data not found error, download " + self.name + " failed"
            )

    def obtain_synthetic_dataset(self, **synthetic_graph_hyperparameters):
        """
        Obtains a synthetic dataset.
        Checks whether the dataset is already generated.
        """
        # Check if the file already exists.
        if osp.exists(self.meta_dict["fname"]):
            print("Synthetic dataset file found, skipping synthetic graph generation.")
            return

        print(
            f"{BColors.WARNING}Synthetic graph generation started, this might take a while . . . {BColors.ENDC}"
        )

        if osp.isdir(self.root):
            print("Dataset directory is ", self.root)
        else:
            os.makedirs(self.root)

        # Generate a synthetic graph.
        src, dst, msg, t, label, idx = generate_synthetic_graph(
            **synthetic_graph_hyperparameters
        )

        # Save the synthetic graph.
        df = pd.DataFrame(
            data={"u": src, "i": dst, "ts": t, "label": label, "idx": idx}
        )
        df.to_csv(self.meta_dict["fname"], index=True)
        np.save(self.meta_dict["fname_edge"], arr=msg)

        print(f"{BColors.OKGREEN}Generation completed {BColors.ENDC}")

    def generate_processed_files(self) -> pd.DataFrame:
        r"""
        Turns raw data .csv file into a pandas data frame, stored on disc if not already.
        Returns:
            df: Pandas data frame.
        """
        node_feat = None
        if not osp.exists(self.meta_dict["fname"]):
            raise FileNotFoundError(f"File not found at {self.meta_dict['fname']}")

        if self.meta_dict["nodefile"] is not None:
            if not osp.exists(self.meta_dict["nodefile"]):
                raise FileNotFoundError(
                    f"File not found at {self.meta_dict['nodefile']}"
                )
        OUT_DF = self.root + "/" + "ml_{}.pkl".format(self.name)
        OUT_EDGE_FEAT = self.root + "/" + "ml_{}.pkl".format(self.name + "_edge")
        if self.meta_dict["nodefile"] is not None:
            OUT_NODE_FEAT = self.root + "/" + "ml_{}.pkl".format(self.name + "_node")

        if (osp.exists(OUT_DF)) and (self.version_passed is True):
            print("Loading processed file")
            df = pd.read_pickle(OUT_DF)
            edge_feat = load_pkl(OUT_EDGE_FEAT)
            if self.meta_dict["nodefile"] is not None:
                node_feat = load_pkl(OUT_NODE_FEAT)

        else:
            print("File not processed, generating processed file.")
            if self.name == "tgbl-flight":
                df, edge_feat, node_ids = csv_to_pd_data(self.meta_dict["fname"])
            elif self.name == "tgbl-coin":
                df, edge_feat, node_ids = csv_to_pd_data_sc(self.meta_dict["fname"])
            elif self.name == "tgbl-comment":
                df, edge_feat, node_ids = csv_to_pd_data_rc(self.meta_dict["fname"])
            elif self.name == "tgbl-review":
                df, edge_feat, node_ids = csv_to_pd_data_sc(self.meta_dict["fname"])
            elif self.name == "tgbl-wiki":
                df, edge_feat, node_ids = load_edgelist_wiki(self.meta_dict["fname"])
            elif self.name in ["synthetic", "lanl", "darpa-trace", "darpa-theia"]:
                df, edge_feat, node_ids = load_preprocessed_data(
                    self.meta_dict["fname"], self.meta_dict["fname_edge"], shift=False
                )
            elif self.custom_dataset:
                df, edge_feat, node_ids = load_preprocessed_data(
                    self.meta_dict["fname"], self.meta_dict["fname_edge"], shift=True
                )

            save_pkl(edge_feat, OUT_EDGE_FEAT)
            df.to_pickle(OUT_DF)
            if self.meta_dict["nodefile"] is not None:
                node_feat = process_node_feat(self.meta_dict["nodefile"], node_ids)
                save_pkl(node_feat, OUT_NODE_FEAT)

        return df, edge_feat, node_feat

    def pre_process(self):
        """
        Pre-process the dataset and generates the splits. It must be run before dataset properties can be accessed.
        Generates the edge data and different train, val, test splits.
        """
        # Check if path to file is valid.
        df, edge_feat, node_feat = self.generate_processed_files()
        sources = np.array(df["u"])
        destinations = np.array(df["i"])
        timestamps = np.array(df["ts"])
        edge_idxs = np.array(df["idx"])
        weights = np.array(df["w"])

        edge_label = np.ones(len(df))  # Should be 1 for all positive edges.
        self._edge_feat = edge_feat
        self._node_feat = node_feat

        full_data = {
            "sources": sources,
            "destinations": destinations,
            "timestamps": timestamps,
            "edge_idxs": edge_idxs,
            "edge_feat": edge_feat,
            "w": weights,
            "edge_label": edge_label,
        }
        self._full_data = full_data
        _train_mask, _val_mask, _test_mask = self.generate_splits(
            full_data, val_ratio=self.val_ratio, test_ratio=self.test_ratio
        )
        self._train_mask = _train_mask
        self._val_mask = _val_mask
        self._test_mask = _test_mask

    def generate_splits(
        self,
        full_data: Dict[str, Any],
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        r"""Generates train, validation, and test splits from the full dataset.
        Args:
            full_data: Dictionary containing the full dataset.
            val_ratio: Ratio of validation data.
            test_ratio: Ratio of test data.
        Returns:
            train_data: Dictionary containing the training dataset.
            val_data: Dictionary containing the validation dataset.
            test_data: Dictionary containing the test dataset.
        """
        val_time, test_time = list(
            np.quantile(
                full_data["timestamps"],
                [(1 - val_ratio - test_ratio), (1 - test_ratio)],
            )
        )
        timestamps = full_data["timestamps"]

        train_mask = timestamps <= val_time
        val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
        test_mask = timestamps > test_time

        return train_mask, val_mask, test_mask

    @property
    def eval_metric(self) -> List[str]:
        """
        The official evaluation metrics for the dataset.
        Returns:
            eval_metric: List[str], The evaluation metrics.
        """
        return self.metric

    def load_eval_set(
        self,
        fname: str,
        split_mode: str = "val",
    ) -> None:
        r"""
        Loads the evaluation set from disk, can be either val or test set anomalous samples.
        Parameters:
            fname: The file name of the evaluation anomalies on disk.
            split_mode: The split mode of the evaluation set, can be either `val` or `test`.

        Returns:
            None
        """
        assert split_mode in [
            "val",
            "test",
        ], "Invalid split-mode! It should be `val`, `test`"
        if not os.path.exists(fname):
            raise FileNotFoundError(f"File not found at {fname}")
        anomalous_samples = load_pkl(fname)
        return anomalous_samples

    def load_anomalies_split(
        self,
        anom_type: str,
        anom_set_id: int,
        split_mode: str,
    ) -> dict:
        r"""
        Load the anomalous samples for an evaluation split.
        Parameters:
            anom_type (str): Type of anomalies to load.
            anom_set_id (int): ID of anomaly set (multiple sets of anomalies are supported).
            split_mode: The split mode of the evaluation set, can be either `val` or `test`.
        """
        if anom_type is None:
            print(f"No anomalies were loaded in {split_mode} set.")
            return {
                key: np.empty(shape=(0,) + self._full_data[key].shape[1:])
                for key in self._full_data
            }

        if anom_type == "organic":
            assert (
                self.name in DATA_ORGANIC_ANOMALIES
            ), f"The {self.name} dataset does not have organic anomalies!"

        if self._anomalies_loaded[split_mode] is not None:
            print(
                f"Can't load {anom_type} anomalies in {split_mode} set because {self._anomalies_loaded[split_mode]} anomalies are already loaded. \
                Skipping this operation."
            )
            return

        assert (
            anom_type in ANOMALY_ABBREVIATIONS
        ), f"The supported types of anomalies are: {list(ANOMALY_ABBREVIATIONS.keys())}!"

        fname = self.meta_dict[f"{split_mode}_anomalies"].replace(
            "{anom_type}", ANOMALY_ABBREVIATIONS[anom_type]
        )
        fname = fname.replace("{anom_set_id}", str(anom_set_id))
        anomalies = self.load_eval_set(fname=fname, split_mode=split_mode)

        # Flag to prevent loading anomalies multiple times
        self._anomalies_loaded[split_mode] = anom_type

        print(
            f"{self._anomalies_loaded[split_mode]} anomalies were successfully loaded in {split_mode} set."
        )
        return anomalies

    def load_anomalies(
        self,
        val_anom_type: str,
        test_anom_type: str,
        anom_set_id: Optional[int] = 0,
    ) -> None:
        r"""
        Load the anomalous samples for the validation and test set.
        Parameters:
            val_anom_type (str): Type of anomalies to load in validation set.
            test_anom_type (str): Type of anomalies to load in test set.
            anom_set_id (Optional[int], optional): ID of anomaly set (multiple sets of anomalies are supported).
                Defaults to 0.
        """
        val_anomalies = self.load_anomalies_split(
            anom_type=val_anom_type,
            anom_set_id=anom_set_id,
            split_mode="val",
        )

        test_anomalies = self.load_anomalies_split(
            anom_type=test_anom_type,
            anom_set_id=anom_set_id,
            split_mode="test",
        )

        # Extend the dataset with anomalous samples.
        for key in self._full_data:
            self._full_data[key] = np.concatenate(
                [self._full_data[key], val_anomalies[key], test_anomalies[key]], axis=0
            )

        # Update all masks according to the new samples.
        num_val_anom = val_anomalies["sources"].shape[0]
        num_test_anom = test_anomalies["sources"].shape[0]
        num_all_anom = num_val_anom + num_test_anom

        anom_train_mask = np.zeros(shape=num_all_anom, dtype=bool)

        anom_val_mask = np.zeros(shape=num_all_anom, dtype=bool)
        anom_val_mask[:num_val_anom] = True

        anom_test_mask = np.zeros(shape=num_all_anom, dtype=bool)
        anom_test_mask[num_val_anom:] = True

        self._train_mask = np.concatenate([self._train_mask, anom_train_mask], axis=0)
        self._val_mask = np.concatenate([self._val_mask, anom_val_mask], axis=0)
        self._test_mask = np.concatenate([self._test_mask, anom_test_mask], axis=0)

        # Sort the samples according to timestamps.
        t = self._full_data["timestamps"]
        perm = np.argsort(t)

        for key in self._full_data:
            self._full_data[key] = self._full_data[key][perm]

        self._train_mask = self._train_mask[perm]
        self._val_mask = self._val_mask[perm]
        self._test_mask = self._test_mask[perm]

        # Read information about edge features.
        self._edge_feat = self._full_data["edge_feat"]

    @property
    def node_feat(self) -> Optional[np.ndarray]:
        r"""
        Returns the node features of the dataset with dim [N, feat_dim].
        Returns:
            node_feat: np.ndarray, [N, feat_dim] or None if there is no node feature.
        """
        return self._node_feat

    @property
    def edge_feat(self) -> Optional[np.ndarray]:
        r"""
        Returns the edge features of the dataset with dim [E, feat_dim].
        Returns:
            edge_feat: np.ndarray, [E, feat_dim] or None if there is no edge feature.
        """
        return self._edge_feat

    @property
    def full_data(self) -> Dict[str, Any]:
        r"""
        The full data of the dataset as a dictionary with keys: 'sources', 'destinations', 'timestamps', 'edge_idxs', 'edge_feat', 'w', 'edge_label'.

        Returns:
            full_data: Dict[str, Any].
        """
        if self._full_data is None:
            raise ValueError(
                "Dataset has not been processed yet, please call pre_process() first."
            )
        return self._full_data

    @property
    def train_mask(self) -> np.ndarray:
        r"""
        Returns the train mask of the dataset.
        Returns:
            train_mask: Training masks.
        """
        if self._train_mask is None:
            raise ValueError("Training split hasn't been loaded.")
        return self._train_mask

    @property
    def val_mask(self) -> np.ndarray:
        r"""
        Returns the validation mask of the dataset.
        Returns:
            val_mask: Dict[str, Any].
        """
        if self._val_mask is None:
            raise ValueError("Validation split hasn't been loaded.")
        return self._val_mask

    @property
    def test_mask(self) -> np.ndarray:
        r"""
        Returns the test mask of the dataset.
        Returns:
            test_mask: Dict[str, Any]
        """
        if self._test_mask is None:
            raise ValueError("Test split hasn't been loaded.")
        return self._test_mask


def main():
    name = "tgbl-wiki"
    dataset = LinkAnomDetDataset(name=name, root="datasets", preprocess=True)

    dataset.node_feat
    dataset.edge_feat  # Not the same as edge weights
    dataset.full_data
    dataset.full_data["edge_idxs"]
    dataset.full_data["sources"]
    dataset.full_data["destinations"]
    dataset.full_data["timestamps"]
    dataset.full_data["edge_label"]


if __name__ == "__main__":
    main()
