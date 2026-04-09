# src/server/__init__.py
from src.server.aggregator import fedavg, fedprox
from src.server.server import FederatedServer

__all__ = ["fedavg", "fedprox", "FederatedServer"]
