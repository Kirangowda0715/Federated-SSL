"""Smoke test for FedSSL pipeline — run this to verify everything works."""
import sys
import copy
import traceback
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("FedSSL Smoke Test")
print("=" * 60)

# 1. Config
from src.utils.config import load_config
config = load_config("configs/default.yaml")
print(f"[OK] Config loaded | backbone={config.model.backbone} | rounds={config.federated.rounds}")

# 2. Metrics
from src.utils.metrics import evaluate, format_metrics
metrics = evaluate(np.array([0,1,0,1]), np.array([0.1,0.9,0.2,0.8]))
print(f"[OK] Metrics: {format_metrics(metrics)}")

# 3. MAE model
from src.models.mae import build_mae
mae = build_mae(backbone="resnet50", embed_dim=512, mask_ratio=0.75, decoder_depth=2)
mae.eval()
x = torch.randn(2, 3, 224, 224)
loss, pred, mask = mae(x)
print(f"[OK] MAE forward | loss={loss.item():.4f} | pred={pred.shape} | mask={mask.shape}")

# 4. Encoder weights
enc_weights = mae.get_encoder_weights()
print(f"[OK] Encoder weights | keys={len(enc_weights)}")

# 5. FedAvg
from src.server.aggregator import fedavg
w1 = mae.get_encoder_weights()
w2 = mae.get_encoder_weights()
agg = fedavg([w1, w2], [100, 200])
print(f"[OK] FedAvg | keys={len(agg)}")

# 6. FederatedServer
from src.server.server import FederatedServer
server = FederatedServer(config)
global_model = server.initialize_global_model()
bcast = server.broadcast()
print(f"[OK] Server initialized + broadcast | broadcast keys={len(bcast)}")

# 7. ssl_local_train (1 epoch, small batch)
from src.client.ssl_train import ssl_local_train

class TinyDataset(torch.utils.data.Dataset):
    def __len__(self): return 8
    def __getitem__(self, i): return torch.randn(3, 224, 224)

from src.utils.config import load_config
cfg2 = load_config("configs/default.yaml")
# Override to 1 epoch for speed
cfg2.ssl.epochs_per_round = 1
cfg2.ssl.batch_size = 4

loader = torch.utils.data.DataLoader(TinyDataset(), batch_size=4)
result = ssl_local_train(
    hospital_id=1,
    model=copy.deepcopy(global_model),
    dataloader=loader,
    config=cfg2,
    global_weights=bcast,
)
print(f"[OK] ssl_local_train | samples={result['num_samples']} | loss={result['epoch_losses'][-1]:.4f}")

# 8. Server aggregate + update
agg2 = server.aggregate([result["encoder_weights"]], [result["num_samples"]])
server.update_global_model(agg2)
print(f"[OK] Server aggregate + update")

# 9. PrototypicalHead
from src.models.proto_head import PrototypicalHead
ph = PrototypicalHead(embed_dim=512, num_classes=2)
supp_emb = torch.randn(4, 512)
supp_lbl = torch.tensor([0, 0, 1, 1])
protos = ph.compute_prototypes(supp_emb, supp_lbl)
query_emb = torch.randn(6, 512)
probs = ph.forward(query_emb, protos)
print(f"[OK] PrototypicalHead | protos={protos.shape} | probs={probs.shape}")

# 10. Checkpoint
ckpt_path = server.save_checkpoint(0, metrics={"auc": 0.87})
print(f"[OK] Checkpoint saved: {ckpt_path}")

print()
print("=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
