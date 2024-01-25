import copy
import time
import timeit
import argparse

from tgb.linkanomdet.anomaly_generator import AnomalousEdgeGenerator
from tgb.linkanomdet.dataset_pyg import PyGLinkAnomDetDataset
from tgb.utils.info import PROJ_DIR


def generate_anomalies(args):
    r"""
    Generate anomalous edges for the validation and test phase.
    """
    print("*** Anomalous Sample Generation ***")

    dataset = PyGLinkAnomDetDataset(
        name=args.dataset_name,
        root=args.output_root,
        absolute_path=True,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    data = dataset.get_TemporalData()

    data_splits = {}
    data_splits["train"] = data[train_mask]
    data_splits["val"] = data[val_mask]
    data_splits["test"] = data[test_mask]

    # Ensure to only sample actual nodes when generating anomalies.
    min_src_idx, max_src_idx = int(data.src.min()), int(data.src.max())
    min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())
    min_idx, max_idx = min(min_src_idx, min_dst_idx), max(max_src_idx, max_dst_idx)
    historical_data = data_splits["train"]

    anom_sampler = AnomalousEdgeGenerator(
        dataset_name=args.dataset_name,
        first_id=min_idx,
        last_id=max_idx,
        anom_prop=args.anom_prop,
        anom_type=args.anom_type,
        anom_set_id=args.anom_set_id,
        rnd_seed=args.seed,
        historical_data=historical_data,
    )

    # Replace hyphens with underlines.
    dataset_folder = args.dataset_name.replace("-", "_")
    partial_path = f"{args.output_root}/{dataset_folder}"
    # Generate anomalous edges for validation set.
    start_time = time.time()
    split_mode = "val"
    print(
        f"INFO: Start generating {args.anom_type} anomalies with ID {args.anom_set_id}: {split_mode}"
    )
    anom_sampler.generate_anomalous_samples(
        data=data_splits[split_mode], split_mode=split_mode, partial_path=partial_path
    )
    print(
        f"INFO: End of anomalous samples generation. Elapsed Time (s): {time.time() - start_time: .4f}"
    )

    # Generate anomalous edges for test set.
    start_time = timeit.default_timer()
    split_mode = "test"
    print(
        f"INFO: Start generating {args.anom_type} anomalies with ID {args.anom_set_id}: {split_mode}"
    )
    anom_sampler.generate_anomalous_samples(
        data=data_splits[split_mode], split_mode=split_mode, partial_path=partial_path
    )
    print(
        f"INFO: End of anomalous samples generation. Elapsed Time (s): {timeit.default_timer()- start_time: .4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Interface for link anomaly generation")
    parser.add_argument(
        "--output_root",
        type=str,
        help="Where to write the output (absolute path).",
        default="/data/shares/stor02/tpostuvan/DATASETS",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset name (e.g., tgbl-wiki or reddit)",
        default="tgbl-wiki",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        help="Ratio of validation data",
        default=0.15,
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        help="Ratio of test data",
        default=0.15,
    )
    parser.add_argument(
        "--anom_type",
        type=str,
        help="Type of anomalies to inject",
        default="all",
        choices=[
            "temporal-structural-contextual",
            "structural-contextual",
            "temporal-contextual",
            "temporal",
            "contextual",
            "combination",
            "all",
        ],
    )
    parser.add_argument(
        "--num_anom_sets",
        type=int,
        help="Number of anomaly sets to generate",
        default=1,
    )
    parser.add_argument(
        "--anom_prop",
        type=float,
        help="Proportion of anomalies to inject",
        default=0.05,
    )
    parser.add_argument("--seed", type=int, default=44, help="Random seed")
    args = parser.parse_args()

    anom_types = []
    if args.anom_type == "all":
        anom_types = [
            "temporal-structural-contextual",
            "structural-contextual",
            "temporal-contextual",
            "temporal",
            "contextual",
            "combination",
        ]
    else:
        anom_types = [args.anom_type]

    for anom_type in anom_types:
        for anom_set_id in range(args.num_anom_sets):
            cur_args = copy.copy(args)
            cur_args.anom_type = anom_type
            cur_args.anom_set_id = anom_set_id
            generate_anomalies(cur_args)
