# FedSSL — Federated Self-Supervised Learning for TB Detection

Complete build of a production-grade FedSSL research system from scratch, implementing MAE-based self-supervised pretraining across 5 simulated hospitals, federated aggregation (FedAvg/FedProx), and prototypical few-shot fine-tuning for TB detection.

## User Review Required

> [!IMPORTANT]
> The existing repo has a **flat structure** (models/, federated/, utils/ at root level) and references **Flower (flwr)** for federated learning. The master prompt specifies a `src/`-nested layout with **custom FedAvg/FedProx** (no Flower). The plan proposes a **fresh layout inside the existing repo** using the `src/` structure, leaving old empty dirs and stub files in place (or overwriting them). Please confirm this is acceptable, or if you'd like those old stubs removed.

> [!WARNING]
> The existing `train_federated.py`, `train_ssl.py`, `evaluate.py`, `preprocessing.py` are stubs (we haven't looked at their contents yet). These WON'T be deleted — they'll remain as legacy stubs unless you ask us to clean them up.

> [!CAUTION]
> The `notebooks/exploration.ipynb` will be built as a **fully pre-coded notebook** with placeholder logic (data may not exist at build time). Cell outputs will be empty until you download the datasets and run the notebook end-to-end.

---

## Proposed Changes

### Component 1 — Config & Utils

#### [NEW] [default.yaml](file:///d:/Projects/Final%20Year%20Project/Federated-SSL/configs/default.yaml)
All hyperparameters: data paths, model backbone, SSL, federated rounds, fine-tuning k-shot, evaluation metrics, logging config.

#### [NEW] [config.py](file:///d:/Projects/Final%20Year%20Project/Federated-SSL/src/utils/config.py)
- `load_config(path)` → returns nested `SimpleNamespace`
- Validates required fields
- Parses CLI `--key.subkey=value` overrides via `argparse`

#### [NEW] [metrics.py](file:///d:/Projects/Final%20Year%20Project/Federated-SSL/src/utils/metrics.py)
- `evaluate(y_true, y_pred_proba)` → dict with AUC, accuracy, sensitivity, specificity, F1, confusion matrix
- Built on `sklearn.metrics`

---

### Component 2 — Dataset Loaders

#### [NEW] [loader.py](file:///d:/Projects/Final%20Year%20Project/Federated-SSL/src/datasets/loader.py)
- `NIHDataset` — unlabeled, SSL two-view augmentation
- `ShenzhenDataset` — TB/Normal binary labels
- `MontgomeryDataset` — TB/Normal binary labels
- Standard chest X-ray transforms (flip, crop, ImageNet normalize)

#### [NEW] [splitter.py](file:///d:/Projects/Final%20Year%20Project/Federated-SSL/src/datasets/splitter.py)
- `split_nih_to_hospitals(dataset, num_hospitals=5, strategy='non_iid')`
- IID: random equal partition
- Non-IID: Dirichlet(α=0.5) distribution per disease label buckets
- Saves `data/processed/hospital_{i}/indices.npy`

---

### Component 3 — Model Architecture

#### [NEW] [encoder.py](file:///d:/Projects/Final%20Year%20Project/Federated-SSL/src/models/encoder.py)
- `get_encoder(backbone, embed_dim)` factory
- ResNet50: removes FC, adds Linear(2048 → embed_dim)
- ViT-Small: via `timm`, replaces head with Linear(384 → embed_dim)

#### [NEW] [decoder.py](file:///d:/Projects/Final%20Year%20Project/Federated-SSL/src/models/decoder.py)
- `MAEDecoder(embed_dim, num_patches, patch_size, decoder_depth=4)`
- Transformer decoder with 4 layers
- Outputs reconstructed pixel values for all patches
- MSE loss computed only on masked patches

#### [NEW] [mae.py](file:///d:/Projects/Final%20Year%20Project/Federated-SSL/src/models/mae.py)
- `MaskedAutoencoder(encoder, decoder, mask_ratio=0.75)`
- `patchify()`, `random_masking()`, `forward()`, `get_encoder_weights()`
- Handles patch embedding for both ResNet50 and ViT backbones

#### [NEW] [proto_head.py](file:///d:/Projects/Final%20Year%20Project/Federated-SSL/src/models/proto_head.py)
- `PrototypicalHead(embed_dim, num_classes=2)`
- `compute_prototypes(support_embeddings, support_labels)`
- `forward(query_embeddings, prototypes)` → softmax over negative L2 distances
- CE fine-tuning fallback

---

### Component 4 — Federated Client

#### [NEW] [ssl_train.py](file:///d:/Projects/Final%20Year%20Project/Federated-SSL/src/client/ssl_train.py)
- `ssl_local_train(hospital_id, model, dataloader, config, global_weights=None)`
- AdamW + cosine LR schedule
- FedProx proximal term: `μ/2 * ||w - w_global||²` added to loss when `global_weights` provided
- Returns only encoder state_dict

#### [NEW] [local_train.py](file:///d:/Projects/Final%20Year%20Project/Federated-SSL/src/client/local_train.py)
- `finetune_local(hospital_id, encoder, shenzhen_loader, config)`
- Freeze encoder, sample k-shot support set
- Compute prototypes → fine-tune on query set
- Returns model + validation metrics dict

---

### Component 5 — Federated Server

#### [NEW] [aggregator.py](file:///d:/Projects/Final%20Year%20Project/Federated-SSL/src/server/aggregator.py)
- `fedavg(encoder_weights_list, sample_counts)` → weighted average state_dict
- `fedprox(global_weights, local_weights_list, sample_counts, mu)` → same as fedavg on server (proximal on client)

#### [NEW] [server.py](file:///d:/Projects/Final%20Year%20Project/Federated-SSL/src/server/server.py)
- `FederatedServer(config)` class
- `initialize_global_model()`, `broadcast()`, `aggregate()`, `update_global_model()`, `save_checkpoint()`
- Tracks best model by Montgomery AUC

---

### Component 6 — Simulation

#### [NEW] [simulation.py](file:///d:/Projects/Final%20Year%20Project/Federated-SSL/src/federated/simulation.py)
- Main CLI entry point: `python src/federated/simulation.py --config configs/default.yaml`
- Full federated loop over N rounds
- Sequential (default) + parallel (multiprocessing) modes
- Every 5 rounds: fine-tune on Shenzhen, evaluate on Montgomery
- Prints per-round summary table with loss + metrics
- Saves round logs to `experiments/logs/`

---

### Component 7 — Requirements & Documentation

#### [MODIFY] [requirements.txt](file:///d:/Projects/Final%20Year%20Project/Federated-SSL/requirements.txt)
Replace stub with pinned versions: torch, torchvision, timm, numpy, scikit-learn, Pillow, pyyaml, matplotlib, seaborn, tqdm, pandas, jupyter.

#### [MODIFY] [README.md](file:///d:/Projects/Final%20Year%20Project/Federated-SSL/README.md)
Full project README: overview, dataset download instructions (NIH/Shenzhen/Montgomery), setup, how to run, expected outputs, ASCII architecture diagram.

---

### Component 8 — Notebook

#### [NEW] [exploration.ipynb](file:///d:/Projects/Final%20Year%20Project/Federated-SSL/notebooks/exploration.ipynb)
Pre-coded Jupyter notebook with 5 sections:
1. EDA — visualize sample X-rays from all 3 datasets
2. MAE masking visualizer
3. t-SNE of encoder embeddings before/after SSL
4. Federated training loss curves
5. Final ROC curve on Montgomery test set

---

## Design Decisions

| Decision | Rationale |
|---|---|
| `src/` layout (not flat) | Matches master prompt spec; cleaner separation of concerns |
| No Flower dependency | Custom FedAvg/FedProx is standard in research papers; Flower adds overhead |
| ResNet50 default | More stable than ViT for limited medical imaging data; ViT still supported |
| Dirichlet α=0.5 | Standard heterogeneity benchmark in FL literature |
| Encoder-only sharing | Core FedSSL privacy principle: decoder stays local |
| FedProx on client | Proximal term must be in client loss, NOT in aggregation |
| `__init__.py` files | Added to all `src/` subpackages for clean imports |

---

## Build Order

Files will be created in dependency order:
1. `configs/default.yaml`
2. `src/utils/config.py` + `src/utils/metrics.py`
3. `src/datasets/loader.py` + `src/datasets/splitter.py`
4. `src/models/encoder.py` → `decoder.py` → `mae.py` → `proto_head.py`
5. `src/client/ssl_train.py` → `local_train.py`
6. `src/server/aggregator.py` → `server.py`
7. `src/federated/simulation.py`
8. `requirements.txt`, `README.md`, `notebooks/exploration.ipynb`
9. All `__init__.py` files

---

## Verification Plan

### Automated
- Run `python src/federated/simulation.py --config configs/default.yaml` with **dummy/synthetic data** (random tensors) to verify the full pipeline without needing actual dataset downloads
- Verify import chain: `python -c "from src.federated.simulation import main"`
- Check shape assertions in MAE forward pass

### Manual
- User downloads datasets (NIH, Shenzhen, Montgomery) and places them in `data/raw/`
- Re-runs simulation with real data to achieve target AUC ≥ 0.85 on Montgomery

---

## Open Questions

> [!IMPORTANT]
> **Q1**: Should the existing flat files (`train_federated.py`, `train_ssl.py`, `evaluate.py`, `preprocessing.py`) be **deleted** to clean up the repo, or **kept** as reference stubs?

> [!IMPORTANT]
> **Q2**: For the **notebook** (`exploration.ipynb`), should it use synthetic dummy data so it can be run without the real datasets, or should it strictly reference the real data paths (requiring dataset downloads first)?

> [!IMPORTANT]
> **Q3**: The `__init__.py` requirement — should I add these to make `src/` a proper Python package (needed for `from src.models.encoder import ...` style imports), or should it use relative imports only?
