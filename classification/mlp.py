import argparse
import copy
import glob
import json
import os
import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_float(x, default=0.0):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default

class JsonFlatNodeDataset(Dataset):
    """
    Fixed-node, node-flattened dataset for MLP.

    Each graph is converted to:
        [node_0_features, node_1_features, ..., node_{N-1}_features]

    Features used per node:
        - id_norm
        - has_orig_id
        - orig_id_norm
        - node_type one-hot: [is_head, is_ffn]
        - layer one-hot (num_layers dims)
        - has_head_idx, head_idx
        - has_ffn_idx, ffn_idx
        - score
        - max_val
        - pos_com
        - pos_ltr
        - max_pos
        - has_pos_enc
        - pos_enc padded/truncated to pos_enc_dim
    """

    def __init__(
        self,
        graph_dir: str,
        max_samples: Optional[int] = None,
        num_layers: Optional[int] = None,
        pos_enc_dim: Optional[int] = None,
    ):
        self.graph_dir = graph_dir
        self.paths = sorted(glob.glob(os.path.join(graph_dir, "*.json")))

        if max_samples is not None:
            self.paths = self.paths[:max_samples]

        if len(self.paths) == 0:
            raise ValueError(f"No JSON graphs found in {graph_dir}")

        print(f"Loading JSON graphs from: {graph_dir}")
        print(f"Found {len(self.paths)} JSON files")

        self.num_nodes = None
        inferred_num_layers = 0
        inferred_pos_enc_dim = 0

        self.labels = []
        self.sample_ids = []

        for path in self.paths:
            with open(path, "r") as f:
                obj = json.load(f)

            nodes = obj.get("nodes", [])
            metadata = obj.get("metadata", {})

            if self.num_nodes is None:
                self.num_nodes = len(nodes)
            elif len(nodes) != self.num_nodes:
                raise ValueError(
                    f"Node count mismatch in {path}: "
                    f"expected {self.num_nodes}, got {len(nodes)}"
                )

            max_layer_here = 0
            max_pos_enc_here = 0

            for n in nodes:
                layer_depth = int(round(safe_float(n.get("layer_depth", 0.0))))
                max_layer_here = max(max_layer_here, layer_depth)

                raw_pos_enc = n.get("pos_enc", [])
                if isinstance(raw_pos_enc, list):
                    max_pos_enc_here = max(max_pos_enc_here, len(raw_pos_enc))

            inferred_num_layers = max(inferred_num_layers, max_layer_here + 1)
            inferred_pos_enc_dim = max(inferred_pos_enc_dim, max_pos_enc_here)

            self.labels.append(int(metadata.get("label", 0)))
            self.sample_ids.append(int(metadata.get("sample_id", len(self.sample_ids))))

        self.num_layers = inferred_num_layers if num_layers is None else num_layers
        self.pos_enc_dim = inferred_pos_enc_dim if pos_enc_dim is None else pos_enc_dim

        if inferred_num_layers > self.num_layers:
            raise ValueError(
                f"Inferred num_layers={inferred_num_layers} from data, "
                f"but requested num_layers={self.num_layers}"
            )

        if inferred_pos_enc_dim > self.pos_enc_dim:
            raise ValueError(
                f"Inferred pos_enc_dim={inferred_pos_enc_dim} from data, "
                f"but requested pos_enc_dim={self.pos_enc_dim}"
            )

        self.per_node_dim = (
            1 + 1 + 1 + 2 + self.num_layers + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + self.pos_enc_dim
        )

        X_list = []
        y_list = []
        sid_list = []

        for path in self.paths:
            with open(path, "r") as f:
                obj = json.load(f)

            x = self._graph_to_feature(obj)
            y = int(obj.get("metadata", {}).get("label", 0))
            sid = int(obj.get("metadata", {}).get("sample_id", 0))

            X_list.append(x)
            y_list.append(y)
            sid_list.append(sid)

        self.X = np.stack(X_list).astype(np.float32) 
        self.y = np.array(y_list, dtype=np.int64)
        self.sample_ids = np.array(sid_list, dtype=np.int64)

        self.mean = None
        self.std = None

        print(f"Fixed num_nodes: {self.num_nodes}")
        print(f"Using num_layers: {self.num_layers}")
        print(f"Using pos_enc_dim: {self.pos_enc_dim}")
        print(f"Per-node feature dim: {self.per_node_dim}")
        print(f"Flattened graph feature dim: {self.X.shape[1]}")

    def _graph_to_feature(self, obj):
        nodes = obj.get("nodes", [])
        nodes = sorted(nodes, key=lambda n: int(n.get("id", 0)))

        if len(nodes) != self.num_nodes:
            raise ValueError(
                f"Graph has {len(nodes)} nodes, expected {self.num_nodes}"
            )

        graph_feat = []

        denom_nodes = max(1, self.num_nodes - 1)

        for n in nodes:
            node_id = int(n.get("id", 0))
            id_norm = node_id / denom_nodes

            raw_orig_id = n.get("orig_id", None)
            has_orig_id = 1.0 if raw_orig_id is not None else 0.0
            orig_id_norm = safe_float(raw_orig_id, 0.0) / denom_nodes if raw_orig_id is not None else 0.0

            node_type = n.get("node_type", "")
            if node_type == "head":
                type_one_hot = [1.0, 0.0]
            elif node_type == "ffn":
                type_one_hot = [0.0, 1.0]
            else:
                type_one_hot = [0.0, 0.0]

           
            layer_depth = int(round(safe_float(n.get("layer_depth", 0.0))))
            layer_depth = max(0, min(self.num_layers - 1, layer_depth))
            layer_one_hot = [0.0] * self.num_layers
            layer_one_hot[layer_depth] = 1.0

          
            raw_head_idx = n.get("head_idx", None)
            has_head_idx = 1.0 if raw_head_idx is not None else 0.0
            head_idx = safe_float(raw_head_idx, 0.0)

            raw_ffn_idx = n.get("neuron_idx", n.get("ffn_idx", None))
            has_ffn_idx = 1.0 if raw_ffn_idx is not None else 0.0
            ffn_idx = safe_float(raw_ffn_idx, 0.0)

           
            score = safe_float(n.get("score", 0.0))
            max_val = safe_float(n.get("max_val", 0.0))
            pos_com = safe_float(n.get("pos_com", 0.0))
            pos_ltr = safe_float(n.get("pos_ltr", 0.0))
            max_pos = safe_float(n.get("max_pos", 0.0))

          
            raw_pos_enc = n.get("pos_enc", [])
            if not isinstance(raw_pos_enc, list):
                raw_pos_enc = []

            has_pos_enc = 1.0 if len(raw_pos_enc) > 0 else 0.0

            pos_enc = [safe_float(v, 0.0) for v in raw_pos_enc[:self.pos_enc_dim]]
            if len(pos_enc) < self.pos_enc_dim:
                pos_enc += [0.0] * (self.pos_enc_dim - len(pos_enc))

            node_feat = (
                [id_norm]
                + [has_orig_id, orig_id_norm]
                + type_one_hot
                + layer_one_hot
                + [has_head_idx, head_idx]
                + [has_ffn_idx, ffn_idx]
                + [score, max_val, pos_com, pos_ltr, max_pos]
                + [has_pos_enc]
                + pos_enc
            )

            graph_feat.extend(node_feat)

        return np.array(graph_feat, dtype=np.float32)

    def apply_normalization(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

        if self.mean.shape[0] != self.X.shape[1]:
            raise ValueError("Normalization mean dimension mismatch")
        if self.std.shape[0] != self.X.shape[1]:
            raise ValueError("Normalization std dimension mismatch")

        self.X = (self.X - self.mean) / self.std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.tensor(self.y[idx], dtype=torch.long)
        sid = torch.tensor(self.sample_ids[idx], dtype=torch.long)
        return x, y, sid


def compute_mean_std(x_train: np.ndarray):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


class FlatNodeMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden1: int = 512,
        hidden2: int = 256,
        hidden3: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(dropout),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(dropout),

            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden3, num_classes),
        )

    def forward(self, x):
        return self.net(x)



def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total = 0
    total_correct = 0

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == y).sum().item()
        total += x.size(0)

    avg_loss = total_loss / max(1, total)
    acc = total_correct / max(1, total)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total = 0

    all_true = []
    all_pred = []
    all_ids = []

    for x, y, sid in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        preds = logits.argmax(dim=-1)

        total_loss += loss.item() * x.size(0)
        total += x.size(0)

        all_true.extend(y.cpu().numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())
        all_ids.extend(sid.cpu().numpy().tolist())

    avg_loss = total_loss / max(1, total)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    acc = (all_true == all_pred).mean()

    prec_bin, rec_bin, f1_bin, _ = precision_recall_fscore_support(
        all_true, all_pred, average="binary", zero_division=0
    )

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_true, all_pred, average="macro", zero_division=0
    )

    cm = confusion_matrix(all_true, all_pred, labels=[0, 1])

    return {
        "loss": avg_loss,
        "acc": acc,
        "prec_bin": prec_bin,
        "rec_bin": rec_bin,
        "f1_bin": f1_bin,
        "prec_macro": prec_macro,
        "rec_macro": rec_macro,
        "f1_macro": f1_macro,
        "cm": cm,
        "true": all_true,
        "pred": all_pred,
        "ids": np.array(all_ids),
    }



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", type=str, required=True,
                        help="Directory containing train JSON graphs")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Directory containing test JSON graphs")

    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_test", type=int, default=None)

    parser.add_argument("--num_layers", type=int, default=None,
                        help="If omitted, infer from train data")
    parser.add_argument("--pos_enc_dim", type=int, default=None,
                        help="If omitted, infer from train data")

    parser.add_argument("--val_ratio", type=float, default=0.15)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden1", type=int, default=512)
    parser.add_argument("--hidden2", type=int, default=256)
    parser.add_argument("--hidden3", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    print("===== CUDA Check =====")
    print("CUDA:", torch.cuda.is_available())
    print("======================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds = JsonFlatNodeDataset(
        args.train_dir,
        max_samples=args.max_train,
        num_layers=args.num_layers,
        pos_enc_dim=args.pos_enc_dim,
    )

    test_ds = JsonFlatNodeDataset(
        args.test_dir,
        max_samples=args.max_test,
        num_layers=train_ds.num_layers,
        pos_enc_dim=train_ds.pos_enc_dim,
    )

    if test_ds.num_nodes != train_ds.num_nodes:
        raise ValueError(
            f"Train/test num_nodes mismatch: train={train_ds.num_nodes}, test={test_ds.num_nodes}"
        )

  
    all_indices = np.arange(len(train_ds))
    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=args.val_ratio,
        random_state=args.seed,
        shuffle=True,
        stratify=train_ds.y,
    )

    print(f"Full training-folder samples: {len(train_ds)}")
    print(f"Train split size: {len(train_idx)}")
    print(f"Validation split size: {len(val_idx)}")
    print("Train split label counts:", np.bincount(train_ds.y[train_idx]))
    print("Validation split label counts:", np.bincount(train_ds.y[val_idx]))
    print("Test label counts:", np.bincount(test_ds.y))

 
    mean, std = compute_mean_std(train_ds.X[train_idx])
    train_ds.apply_normalization(mean, std)
    test_ds.apply_normalization(mean, std)

    train_subset = Subset(train_ds, train_idx.tolist())
    val_subset = Subset(train_ds, val_idx.tolist())

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    num_classes = int(train_ds.y.max() + 1)
    input_dim = train_ds.X.shape[1]

    print(f"Test samples: {len(test_ds)}")
    print(f"Num classes: {num_classes}")
    print(f"Num nodes per graph: {train_ds.num_nodes}")
    print(f"Num layers (one-hot): {train_ds.num_layers}")
    print(f"pos_enc_dim: {train_ds.pos_enc_dim}")
    print(f"Per-node feature dim: {train_ds.per_node_dim}")
    print(f"Flattened graph input dim: {input_dim}")

   
    model = FlatNodeMLP(
        input_dim=input_dim,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        hidden3=args.hidden3,
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_metric = -1.0
    best_state = None
    best_epoch = -1

 
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_metrics = evaluate(model, val_loader, criterion, device)

        if val_metrics["f1_macro"] > best_metric:
            best_metric = val_metrics["f1_macro"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        print(
            f"Ep {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.3f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['acc']:.3f} | "
            f"Val Bin-F1: {val_metrics['f1_bin']:.3f} | "
            f"Val Macro-F1: {val_metrics['f1_macro']:.3f} | "
            f"Best Val Macro-F1: {best_metric:.3f} (Ep {best_epoch})"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val = evaluate(model, val_loader, criterion, device)
    final_test = evaluate(model, test_loader, criterion, device)

    print("\n=== Final Validation Metrics (Best Model) ===")
    print(f"Accuracy:         {final_val['acc']:.4f}")
    print(f"Binary Precision: {final_val['prec_bin']:.4f}")
    print(f"Binary Recall:    {final_val['rec_bin']:.4f}")
    print(f"Binary F1:        {final_val['f1_bin']:.4f}")
    print(f"Macro Precision:  {final_val['prec_macro']:.4f}")
    print(f"Macro Recall:     {final_val['rec_macro']:.4f}")
    print(f"Macro F1:         {final_val['f1_macro']:.4f}")

    print("\n=== Final Test Metrics (Best by Validation Macro-F1) ===")
    print(f"Accuracy:         {final_test['acc']:.4f}")
    print(f"Binary Precision: {final_test['prec_bin']:.4f}")
    print(f"Binary Recall:    {final_test['rec_bin']:.4f}")
    print(f"Binary F1:        {final_test['f1_bin']:.4f}")
    print(f"Macro Precision:  {final_test['prec_macro']:.4f}")
    print(f"Macro Recall:     {final_test['rec_macro']:.4f}")
    print(f"Macro F1:         {final_test['f1_macro']:.4f}")

    print("\nTest Confusion Matrix:")
    print(final_test["cm"])

    print("\nTest Classification Report:")
    print(
        classification_report(
            final_test["true"],
            final_test["pred"],
            zero_division=0
        )
    )

 
    os.makedirs("results", exist_ok=True)

    train_name = os.path.basename(os.path.normpath(args.train_dir))
    test_name = os.path.basename(os.path.normpath(args.test_dir))

    pred_df = pd.DataFrame({
        "sample_id": final_test["ids"],
        "true_label": final_test["true"],
        "predicted_label": final_test["pred"],
    })

    csv_name = f"results_mlp_flat_{train_name}_to_{test_name}.csv"
    csv_path = os.path.join("results", csv_name)
    pred_df.to_csv(csv_path, index=False)
    print(f"\nSaved predictions CSV -> {csv_path}")

    ckpt = {
        "model_state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
        "num_nodes": train_ds.num_nodes,
        "num_layers": train_ds.num_layers,
        "pos_enc_dim": train_ds.pos_enc_dim,
        "per_node_dim": train_ds.per_node_dim,
        "input_dim": input_dim,
        "args": vars(args),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_metric,
        "train_indices": train_idx,
        "val_indices": val_idx,
    }

    model_name = f"mlp_flat_{train_name}_to_{test_name}.pt"
    model_path = os.path.join("results", model_name)
    torch.save(ckpt, model_path)
    print(f"Saved model checkpoint -> {model_path}")


if __name__ == "__main__":
    main()


