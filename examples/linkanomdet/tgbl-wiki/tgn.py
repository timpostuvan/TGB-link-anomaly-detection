"""
Link Anomaly Detection with a TGN model with Early Stopping.

Command for an example run:
    python examples/linkanomdet/tgbl-wiki/tgn.py  \
    --root_dir <OUTPUT-DIR>  \
    --anom_type temporal-structural-contextual  \
    --num_run 1  \
    --seed 1
"""

import math
import timeit

import os
import os.path as osp
from pathlib import Path
import numpy as np

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear

from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader

from torch_geometric.nn import TransformerConv

# Internal imports
from tgb.utils.utils import get_args, set_random_seed, save_results
from tgb.linkanomdet.evaluate import Evaluator
from modules.decoder import LinkPredictor
from modules.emb_module import GraphAttentionEmbedding
from modules.msg_func import IdentityMessage
from modules.msg_agg import LastAggregator
from modules.neighbor_loader import LastNeighborLoader
from modules.memory_module import TGNMemory
from modules.early_stopping import EarlyStopMonitor
from tgb.linkanomdet.dataset_pyg import PyGLinkAnomDetDataset


# ==========
# ========== Define helper functions.
# ==========


def train():
    r"""
    Training procedure for TGN model.
    This function uses some objects that are globally defined in the current scrips.

    Parameters:
        None
    Returns:
        None
    """

    model["memory"].train()
    model["gnn"].train()
    model["link_pred"].train()

    model["memory"].reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # Sample negative destination nodes.
        neg_dst = torch.randint(
            min_dst_idx,
            max_dst_idx + 1,
            (src.size(0),),
            dtype=torch.long,
            device=device,
        )

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = model["memory"](n_id)
        z = model["gnn"](
            z,
            last_update,
            edge_index,
            data.t[e_id].to(device),
            data.msg[e_id].to(device),
        )

        pos_out = model["link_pred"](z[assoc[src]], z[assoc[pos_dst]])
        neg_out = model["link_pred"](z[assoc[src]], z[assoc[neg_dst]])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        model["memory"].update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        model["memory"].detach()
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events


@torch.no_grad()
def test(loader):
    r"""
    Evaluate the dynamic link anomaly detection.

    Parameters:
        loader: An object containing positive as well as anomalous edges of the evaluation set.
    Returns:
        perf_metric: The result of the performance evaluation.
    """
    model["memory"].eval()
    model["gnn"].eval()
    model["link_pred"].eval()

    y_preds = []
    y_labels = []
    for batch in loader:
        src, dst, t, msg, y_label = (
            batch.src,
            batch.dst,
            batch.t,
            batch.msg,
            batch.y,
        )

        n_id = torch.cat([src, dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = model["memory"](n_id)
        z = model["gnn"](
            z,
            last_update,
            edge_index,
            data.t[e_id].to(device),
            data.msg[e_id].to(device),
        )

        y_pred = model["link_pred"](z[assoc[src]], z[assoc[dst]])
        y_preds.append(y_pred.cpu())
        y_labels.append(y_label.cpu())

        # Update memory and neighbor loader with all states (including those belonging to anomalous samples).
        model["memory"].update_state(src, dst, t, msg)
        neighbor_loader.insert(src, dst)

    # Compute AUC.
    input_dict = {
        "y_pred": torch.cat(y_preds),
        "y_label": torch.cat(y_labels),
        "eval_metric": [metric],
    }
    perf_metrics = evaluator.eval(input_dict)[metric]
    return perf_metrics


# ==========
# ==========
# ==========


# Start.
start_overall = timeit.default_timer()

# ========== Set parameters.
args, _ = get_args()
print("INFO: Arguments:", args)

DATA = "tgbl-wiki"
ROOT_DIR = args.root_dir
LR = args.lr
BATCH_SIZE = args.bs
K_VALUE = args.k_value
NUM_EPOCH = args.num_epoch
SEED = args.seed
MEM_DIM = args.mem_dim
TIME_DIM = args.time_dim
EMB_DIM = args.emb_dim
TOLERANCE = args.tolerance
PATIENCE = args.patience
NUM_RUNS = args.num_run
NUM_NEIGHBORS = 10
ANOMALY_TYPE = args.anom_type


MODEL_NAME = "TGN"
# ==========

# Set the device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading
dataset = PyGLinkAnomDetDataset(
    name=DATA,
    root=ROOT_DIR,
    absolute_path=True,
    val_ratio=0.15,
    test_ratio=0.15,
)
dataset.load_anomalies(val_anom_type=ANOMALY_TYPE, test_anom_type=ANOMALY_TYPE)

train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask
data = dataset.get_TemporalData()
data = data.to(device)
metric = "auc"

train_data = data[train_mask]
val_data = data[val_mask]
test_data = data[test_mask]

train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

# Neighhorhood sampler
neighbor_loader = LastNeighborLoader(data.num_nodes, size=NUM_NEIGHBORS, device=device)

# Define the model end-to-end.
memory = TGNMemory(
    data.num_nodes,
    data.msg.size(-1),
    MEM_DIM,
    TIME_DIM,
    message_module=IdentityMessage(data.msg.size(-1), MEM_DIM, TIME_DIM),
    aggregator_module=LastAggregator(),
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=MEM_DIM,
    out_channels=EMB_DIM,
    msg_dim=data.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)

link_pred = LinkPredictor(in_channels=EMB_DIM).to(device)

model = {"memory": memory, "gnn": gnn, "link_pred": link_pred}

optimizer = torch.optim.Adam(
    set(model["memory"].parameters())
    | set(model["gnn"].parameters())
    | set(model["link_pred"].parameters()),
    lr=LR,
)
criterion = torch.nn.BCEWithLogitsLoss()

# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)


print("==========================================================")
print(f"=================*** {MODEL_NAME}: LinkAnomDet: {DATA} ***=============")
print("==========================================================")

evaluator = Evaluator()

# For saving the results
results_path = f"{osp.dirname(osp.abspath(__file__))}/saved_results"
if not osp.exists(results_path):
    os.mkdir(results_path)
    print("INFO: Create directory {}".format(results_path))
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f"{results_path}/{MODEL_NAME}_{DATA}_results.json"

for run_idx in range(NUM_RUNS):
    print("--------------------------------------------------------------------")
    print(f"INFO: >>>>> Run: {run_idx} <<<<<")
    start_run = timeit.default_timer()

    # Set the seed for deterministic results.
    torch.manual_seed(run_idx + SEED)
    set_random_seed(run_idx + SEED)

    # Define an early stopper.
    save_model_dir = f"{osp.dirname(osp.abspath(__file__))}/saved_models/"
    save_model_id = f"{MODEL_NAME}_{DATA}_{SEED}_{run_idx}"
    early_stopper = EarlyStopMonitor(
        save_model_dir=save_model_dir,
        save_model_id=save_model_id,
        tolerance=TOLERANCE,
        patience=PATIENCE,
    )

    # ==================================================== Train & Validation

    val_perf_list = []
    start_train_val = timeit.default_timer()
    for epoch in range(1, NUM_EPOCH + 1):
        # Training
        start_epoch_train = timeit.default_timer()
        loss = train()
        print(
            f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Training elapsed Time (s): {timeit.default_timer() - start_epoch_train: .4f}"
        )

        # Validation
        start_val = timeit.default_timer()
        perf_metric_val = test(val_loader)
        print(f"\tValidation {metric}: {perf_metric_val: .4f}")
        print(
            f"\tValidation: Elapsed time (s): {timeit.default_timer() - start_val: .4f}"
        )
        val_perf_list.append(perf_metric_val)

        # Check for early stopping
        if early_stopper.step_check(perf_metric_val, model):
            break

    train_val_time = timeit.default_timer() - start_train_val
    print(f"Train & Validation: Elapsed Time (s): {train_val_time: .4f}")

    # ==================================================== Test
    # First, load the best model.
    early_stopper.load_checkpoint(model)

    # Final testing
    start_test = timeit.default_timer()
    perf_metric_test = test(test_loader)

    print(f"INFO: Test")
    print(f"\tTest: {metric}: {perf_metric_test: .4f}")
    test_time = timeit.default_timer() - start_test
    print(f"\tTest: Elapsed Time (s): {test_time: .4f}")

    save_results(
        {
            "model": MODEL_NAME,
            "data": DATA,
            "run": run_idx,
            "seed": SEED,
            f"val {metric}": val_perf_list,
            f"test {metric}": perf_metric_test,
            "test_time": test_time,
            "tot_train_val_time": train_val_time,
        },
        results_filename,
    )

    print(
        f"INFO: >>>>> Run: {run_idx}, elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<"
    )
    print("--------------------------------------------------------------------")

print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
print("==============================================================")
