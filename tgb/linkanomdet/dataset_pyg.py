import torch
from typing import Optional, Optional, Callable

from torch_geometric.data import Dataset, TemporalData
from tgb.linkanomdet.dataset import LinkAnomDetDataset
import warnings


class PyGLinkAnomDetDataset(Dataset):
    def __init__(
        self,
        name: str,
        root: str,
        absolute_path: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        val_ratio: Optional[float] = 0.15,
        test_ratio: Optional[float] = 0.15,
    ):
        r"""
        PyG wrapper for the LinkAnomDetDataset.
        It can return pytorch tensors for src, dst, t, msg, label.
        It can return Temporal Data object.
        Parameters:
            name: Name of the dataset, passed to `LinkAnomDetdDataset`.
            root (string): Root directory where the dataset should be saved, passed to `LinkAnomDetdDataset`.
            transform (callable, optional): A function/transform that takes in an, not used in this case.
            pre_transform (callable, optional): A function/transform that takes in, not used in this case.
            val_ratio (float, optional): Ratio of validation data.
            test_ratio (float, optional): Ratio of test data.
        """
        self.name = name
        self.root = root
        self.dataset = LinkAnomDetDataset(
            name=name,
            root=root,
            absolute_path=absolute_path,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        self._train_mask = torch.from_numpy(self.dataset.train_mask)
        self._val_mask = torch.from_numpy(self.dataset.val_mask)
        self._test_mask = torch.from_numpy(self.dataset.test_mask)
        super().__init__(root, transform, pre_transform)
        self._node_feat = self.dataset.node_feat

        if self._node_feat is None:
            self._node_feat = None
        else:
            self._node_feat = torch.from_numpy(self._node_feat).float()
        self.process_data()

    @property
    def eval_metric(self) -> str:
        """
        The official evaluation metric for the dataset.
        Returns:
            eval_metric: str, The evaluation metric.
        """
        return self.dataset.eval_metric

    def load_anomalies(
        self,
        val_anom_type: str,
        test_anom_type: str,
        anom_set_id: Optional[int] = 0,
    ) -> None:
        r"""
        Load the anomalous samples for the validation and test sets.
        Parameters:
            val_anom_type (str): Type of anomalies to load in validation set.
            test_anom_type (str): Type of anomalies to load in test set.
            anom_set_id (Optional[int], optional): ID of anomaly set (multiple sets of anomalies are supported).
                Defaults to 0.
        """
        self.dataset.load_anomalies(
            val_anom_type=val_anom_type,
            test_anom_type=test_anom_type,
            anom_set_id=anom_set_id,
        )

        # Update attributes that are saved explicitly in this class.
        self._train_mask = torch.from_numpy(self.dataset.train_mask)
        self._val_mask = torch.from_numpy(self.dataset.val_mask)
        self._test_mask = torch.from_numpy(self.dataset.test_mask)
        self.process_data()

    @property
    def train_mask(self) -> torch.Tensor:
        r"""
        Returns the train mask of the dataset.
        Returns:
            train_mask: The mask for edges in the training set.
        """
        if self._train_mask is None:
            raise ValueError("training split hasn't been loaded")
        return self._train_mask

    @property
    def val_mask(self) -> torch.Tensor:
        r"""
        Returns the validation mask of the dataset.
        Returns:
            val_mask: The mask for edges in the validation set.
        """
        if self._val_mask is None:
            raise ValueError("validation split hasn't been loaded")
        return self._val_mask

    @property
    def test_mask(self) -> torch.Tensor:
        r"""
        Returns the test mask of the dataset.
        Returns:
            test_mask: The mask for edges in the test set.
        """
        if self._test_mask is None:
            raise ValueError("test split hasn't been loaded")
        return self._test_mask

    @property
    def node_feat(self) -> torch.Tensor:
        r"""
        Returns the node features of the dataset.
        Returns:
            node_feat: The node features.
        """
        return self._node_feat

    @property
    def src(self) -> torch.Tensor:
        r"""
        Returns the source nodes of the dataset.
        Returns:
            src: The idx of the source nodes.
        """
        return self._src

    @property
    def dst(self) -> torch.Tensor:
        r"""
        Returns the destination nodes of the dataset.
        Returns:
            dst: The idx of the destination nodes.
        """
        return self._dst

    @property
    def ts(self) -> torch.Tensor:
        r"""
        Returns the timestamps of the dataset.
        Returns:
            ts: The timestamps of the edges.
        """
        return self._ts

    @property
    def edge_feat(self) -> torch.Tensor:
        r"""
        Returns the edge features of the dataset.
        Returns:
            edge_feat: The edge features.
        """
        return self._edge_feat

    @property
    def edge_label(self) -> torch.Tensor:
        r"""
        Returns the edge labels of the dataset.
        Returns:
            edge_label: The labels of the edges.
        """
        return self._edge_label

    def process_data(self) -> None:
        r"""
        Convert the numpy arrays from dataset to pytorch tensors.
        """
        src = torch.from_numpy(self.dataset.full_data["sources"])
        dst = torch.from_numpy(self.dataset.full_data["destinations"])
        ts = torch.from_numpy(self.dataset.full_data["timestamps"])
        msg = torch.from_numpy(
            self.dataset.full_data["edge_feat"]
        )  # Use edge features here if available.
        edge_label = torch.from_numpy(
            self.dataset.full_data["edge_label"]
        )  # This is the label indicating if an edge is a true edge, 1 for true edges and 0 for anomalies.

        # First check typing for all tensors.
        # Source tensor must be of type int64.
        # warnings.warn("sources tensor is not of type int64 or int32, forcing conversion")
        if src.dtype != torch.int64:
            src = src.long()

        # Destination tensor must be of type int64.
        if dst.dtype != torch.int64:
            dst = dst.long()

        # Timestamp tensor must be of type int64.
        if ts.dtype != torch.int64:
            ts = ts.long()

        # Message tensor must be of type float32.
        if msg.dtype != torch.float32:
            msg = msg.float()

        self._src = src
        self._dst = dst
        self._ts = ts
        self._edge_label = edge_label
        self._edge_feat = msg

    def get_TemporalData(self) -> TemporalData:
        """
        Return the TemporalData object for the entire dataset.
        """
        data = TemporalData(
            src=self._src,
            dst=self._dst,
            t=self._ts,
            msg=self._edge_feat,
            y=self._edge_label,
        )
        return data

    def len(self) -> int:
        """
        Size of the dataset.
        Returns:
            size: int.
        """
        return self._src.shape[0]

    def get(self, idx: int) -> TemporalData:
        """
        Construct temporal data object for a single edge.
        Parameters:
            idx: Index of the edge.
        Returns:
            data: TemporalData object
        """
        data = TemporalData(
            src=self._src[idx],
            dst=self._dst[idx],
            t=self._ts[idx],
            msg=self._edge_feat[idx],
            y=self._edge_label[idx],
        )
        return data

    def __repr__(self) -> str:
        return f"{self.name.capitalize()}()"
