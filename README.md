# Functional Brain Network Diagnosis with Graph Neural Networks (GNNs)

This repository contains the code for **Shayan Bali’s B.Sc. thesis**: _Functional Brain Network Analysis for Disease Diagnosis using Graph Neural Networks_.  
It builds and evaluates multiple **GNN architectures** with **different pooling strategies** on fMRI-derived functional brain networks.

## Highlights
- Implements configurable GNN backbones: `ChebConv, GATConv, GINConv` (via PyTorch Geometric)
- Supports pooling strategies: `ASAPooling, EdgePooling, SAGPooling, TopKPooling` (e.g., hierarchical/Top-K, SAG, ASA, Edge) with the help binary code to benchmark different combination pooling configuration.
- Trains and validates on **HCP-based NeuroGraph** datasets (e.g., `HCPGender`, `HCPActivity`)
- Stratified train/val/test splits with early stopping
- Exports metrics (accuracy, precision, recall, F1), confusion matrix, and training curves

## Repository Structure
- `Shayan_Bali_Thesis_Code.ipynb` — main notebook to mount data, configure experiments, train, evaluate, and plot/export results.
- Outputs are saved under:
  - `results/` — CSV summaries (e.g., `results_new.csv`)
  - `plots/` — training curves and confusion matrix images (filenames contain the run title)

## Data
The notebook uses a `NeuroGraphDataset` loader (imported as `from datasets import NeuroGraphDataset`) with a configurable `root` directory (default `data/`), and expects graphs built from **fMRI** time series. Supported dataset names in the notebook include:
- `HCPGender` (gender classification)
- `HCPActivity` (task/activity classification)

> Place/prepare datasets under `data/` so that `NeuroGraphDataset(root="data/", name=<dataset>)` can load them.

## Requirements
- Python 3.9+
- PyTorch + PyTorch Geometric
- TorchMetrics, scikit-learn, NumPy, Matplotlib, Seaborn
- (Optional) Google Colab for Drive mounting

Minimal install (CPU example):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
pip install torchmetrics scikit-learn matplotlib seaborn
```

> For CUDA builds, install the matching PyTorch & PyG wheels for your CUDA version (see their official instructions).

## Quickstart

### Option A — Run in Google Colab
1. Upload this notebook to Colab.
2. Ensure your data is available under `data/` (or adjust `root` in the notebook).
3. Run all cells.

### Option B — Run Locally (Jupyter)
1. Create and activate a virtual environment; install dependencies.
2. Put datasets under `data/`.
3. Launch Jupyter, open `Shayan_Bali_Thesis_Code.ipynb`, and run cells.

## Configuration (command-style arguments in the notebook)
The notebook uses an `argparse`-style config. Key parameters and defaults:

```text
--dataset          HCPActivity        # also supports HCPGender
--device           cuda               # or cpu
--seed             123
--model            GCNConv            # e.g., ChebConv, GATConv, GINConv
--hidden           32
--hidden_mlp       64
--num_layers       7
--epochs           100
--echo_epoch       50
--batch_size       16
--early_stopping   50                 # patience
--lr               1e-5
--weight_decay     5e-4
--dropout          0.5
--pooling_code     0000000            # per-layer pooling on/off mask
--pooling_type     TopKPooling        # e.g., ASAPooling, EdgePooling, SAGPooling, TopKPooling
```

### Notes
- **Pooling control**: `pooling_code` is a binary string (length ≈ `num_layers`) that selects at which conv layers a pooling module is applied.
- **Model selection**: set `--model` to switch the convolution operator (e.g., `GCNConv`, `GINConv`, `ChebConv`, `GATConv`).
- **Splits**: The code performs stratified 80/10/10 (train/val/test) splitting.
- **Early stopping**: custom `EarlyStopping` class monitors validation loss with a configurable patience.

## Training & Evaluation
- Training loop logs loss and accuracy; evaluation computes accuracy, precision, recall, F1, and a confusion matrix.
- After training, utility functions export:
  - CSV summary to `results/`
  - Plots to `plots/`:
    - `train_loss_history_<title>.png`
    - `val_acc_history_<title>.png`
    - `test_acc_history_<title>.png`
    - `val_loss_history_<title>.png`
    - `confusion_matrix_<title>.png`

## Example Run (inside the notebook)
```python
args.dataset = "HCPGender"
args.model = "GINConv"
args.pooling_type = "TopKPooling"
args.pooling_code = "0101010"   # apply pooling at layers 2,4,6
args.hidden = 64
args.num_layers = 7
args.epochs = 150
args.device = "cuda"            # or "cpu"

# then run the training / evaluation cells
```

## Results
The notebook prints and saves metrics per split. You can aggregate multiple runs in `results/results_new.csv` for comparison across models/pooling settings.

## Acknowledgments
- Built with **PyTorch** and **PyTorch Geometric**.
- Uses an in-notebook `NeuroGraphDataset` interface for HCP-based graph datasets.
- Part of the B.Sc. thesis by **Shayan Bali** (Amirkabir University of Technology, Tehran Polytechnic).
