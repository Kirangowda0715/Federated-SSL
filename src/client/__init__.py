# src/client/__init__.py
from src.client.ssl_train import ssl_local_train
from src.client.local_train import finetune_local, evaluate_on_montgomery

__all__ = ["ssl_local_train", "finetune_local", "evaluate_on_montgomery"]
