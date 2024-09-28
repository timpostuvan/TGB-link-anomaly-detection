# Temporal Graph Benchmark With Link Anomaly Detection Task

## Overview
This repository contains code for link anomaly detection task, integrated into Temporal Graph Benchmark (TGB).
The code is based on [TGB repository](https://github.com/shenyangHuang/TGB).


## Install Dependencies
Our implementation works with python >= 3.9 and can be installed as follows:

1. Set up a [conda](https://docs.conda.io/projects/conda/en/latest/index.html) environment.
```
conda create -n tgb_env python=3.9
conda activate tgb_env
```

2. Install external packages.
```
pip install pandas==1.5.3
pip install matplotlib==3.7.1
pip install clint==0.5.1
```

Install Pytorch and PyG dependencies to run the examples.
```
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu117
pip install torch_geometric==2.3.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

3. Install local dependencies under the root directory `/TGB-link-anomaly-detection`.
```
pip install -e .
```


## Generating Anomalies
The code for generation of anomalies can be found in `/tgb/datasets/dataset_scripts/link_anomaly_generator.py`.

For example, to generate temporal-structural-contextual anomalies for the Wikipedia dataset from TGB, run the following command inside `/tgb/datasets/dataset_scripts` directory:
```
python link_anomaly_generator.py  \
--dataset_name tgbl-wiki  \
--val_ratio 0.15  \
--test_ratio 0.15  \
--anom_type temporal-structural-contextual  \
--output_root <OUTPUT-DIR>
```

The anomalies are generated for the validation and test splits according to the 70/15/15 data split. The data is saved under `<OUTPUT-DIR>` directory, which should be specified as an absolute path.


## Running Example Methods
- For the link anomaly detection task, see the [`examples/linkanomdet`](https://github.com/timpostuvan/TGB-link-anomaly-detection/tree/main/examples/linkanomdet) directory for an example script to run TGN model on a TGB dataset. Note that the example requires generated anomalies, which can be obtained by running the example command in Generating Anomalies section.
- For all other baselines, please see the [CTDG-link-anomaly-detection](https://github.com/timpostuvan/CTDG-link-anomaly-detection) repository.
