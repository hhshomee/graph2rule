
import argparse
import copy
import glob
import json
import os
import random
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Subset
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GatedGraphConv, GCNConv, GATConv, GINConv, global_mean_pool

def materialize_pruned(ds: Dataset, desc: str = ""):
    print(
        f">>> Materializing pruned graphs for {desc} -> "
        f"{getattr(ds, 'derived_pruned_dir', 'UNKNOWN')}",
        flush=True,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    n = 0
    for _ in loader:
        n += 1
        if n % 200 == 0:
            print(f"    {desc}: processed {n}/{len(ds)}", flush=True)
    print(f">>> Done materializing {desc}: {n} graphs", flush=True)


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


class JsonGraphDataset(Dataset):
    def __init__(
        self,
        graphs_dir: str,
        make_undirected: bool = False,
        max_samples: Optional[int] = None,
        prune: bool = False,
        node_ratio: float = 0.5,
        edge_prune: bool = False,
        edge_ratio: float = 0.2,
        num_layers: Optional[int] = None,
        pos_enc_dim: Optional[int] = None,
        num_heads: Optional[int] = None,  
        d_ff: Optional[int] = None,       
    ):
        super().__init__()

        paths = sorted(glob.glob(os.path.join(graphs_dir, "*.json")))
        if max_samples is not None:
            paths = paths[:max_samples]

        if len(paths) == 0:
            raise ValueError(f"No JSON graphs found in {graphs_dir}")

        self.graph_paths = paths
        self._data_cache = [None] * len(self.graph_paths)

        self.make_undirected = make_undirected
        self.prune = prune
        self.node_ratio = node_ratio
        self.edge_prune = edge_prune
        self.edge_ratio = edge_ratio
        self.derived_pruned_dir = graphs_dir.rstrip("/") + "_pruned"
        self.save_enabled = True
        self.do_any_pruning = bool(self.prune or self.edge_prune)

        print(f"Loading graphs from: {graphs_dir}", flush=True)
        print(f"Found {len(paths)} JSON files", flush=True)

        inferred_num_layers = 0
        inferred_pos_enc_dim = 0
        inferred_num_nodes = None
        inferred_max_head_idx = 0  
        inferred_max_ffn_idx = 0   

        self.labels = []
        self.sample_ids = []

        for path in self.graph_paths:
            with open(path, "r") as f:
                obj = json.load(f)

            nodes = obj.get("nodes", [])
            metadata = obj.get("metadata", {})

            if inferred_num_nodes is None:
                inferred_num_nodes = len(nodes)
            elif len(nodes) != inferred_num_nodes:
                print(
                    f"[Warning] Node count mismatch in {path}: "
                    f"expected {inferred_num_nodes}, got {len(nodes)}",
                    flush=True,
                )

            for n in nodes:
                layer_depth = int(round(safe_float(n.get("layer_depth", 0.0))))
                inferred_num_layers = max(inferred_num_layers, layer_depth + 1)

                raw_pos_enc = n.get("pos_enc", [])
                if isinstance(raw_pos_enc, list):
                    inferred_pos_enc_dim = max(inferred_pos_enc_dim, len(raw_pos_enc))

               
                if n.get("node_type") == "head":
                    h = int(round(safe_float(n.get("head_idx", 0))))
                    inferred_max_head_idx = max(inferred_max_head_idx, h)
                elif n.get("node_type") == "ffn":
                    raw = n.get("neuron_idx", n.get("ffn_idx", 0))
                    k = int(round(safe_float(raw, 0)))
                    inferred_max_ffn_idx = max(inferred_max_ffn_idx, k)

            self.labels.append(int(metadata.get("label", 0)))
            self.sample_ids.append(int(metadata.get("sample_id", len(self.sample_ids))))

        self.labels = np.array(self.labels, dtype=np.int64)
        self.sample_ids = np.array(self.sample_ids, dtype=np.int64)

        self.num_nodes = inferred_num_nodes if inferred_num_nodes is not None else 0
        self.num_layers = inferred_num_layers if num_layers is None else num_layers
        self.pos_enc_dim = inferred_pos_enc_dim if pos_enc_dim is None else pos_enc_dim

       
        self.num_heads = (inferred_max_head_idx + 1) if num_heads is None else num_heads
        self.d_ff = (inferred_max_ffn_idx + 1) if d_ff is None else d_ff

        with open(self.graph_paths[0], "r") as f:
            first_obj = json.load(f)
        first_nodes = sorted(first_obj["nodes"], key=lambda n: int(n.get("id", 0)))
        if len(first_nodes) == 0:
            raise ValueError("First graph has zero nodes.")
        self.in_dim = len(self._node_to_feature(first_nodes[0]))

        print(f"Inferred num_layers: {self.num_layers}", flush=True)
        print(f"Inferred pos_enc_dim: {self.pos_enc_dim}", flush=True)
        print(f"Inferred num_heads (for norm): {self.num_heads}", flush=True) 
        print(f"Inferred d_ff (for norm):      {self.d_ff}", flush=True)     
        print(f"Node feature dimension (in_dim): {self.in_dim}", flush=True)

    def len(self):
        return len(self.graph_paths)

    def preload(self, desc: str = "TRAIN", report_every: int = 100):
        """Eagerly materialize this dataset into the in-memory Data cache.

        This performs the expensive JSON -> PyG conversion once before
        training. With DataLoader(num_workers=0), the same cached Data objects
        are then reused by train/validation loaders for every epoch.
        """
        print(
            f">>> Preloading {desc} graphs into CPU RAM cache "
            f"({len(self)} graphs)...",
            flush=True,
        )
        t0 = time.perf_counter()

        for i in range(len(self)):
            _ = self.get(i)
            if report_every > 0 and (i + 1) % report_every == 0:
                print(
                    f"    {desc}: cached {i + 1}/{len(self)} graphs",
                    flush=True,
                )

        elapsed = time.perf_counter() - t0
        cached = sum(x is not None for x in self._data_cache)
        print(
            f">>> Finished preloading {desc}: {cached}/{len(self)} graphs "
            f"cached in {elapsed:.2f} s",
            flush=True,
        )

    def save_json(self, sample_id, nodes, src, dst, wts, label, out_dir):
        os.makedirs(out_dir, exist_ok=True)

        json_edges = []
        for s, d, w in zip(src, dst, wts):
            json_edges.append({
                "source": int(s),
                "target": int(d),
                "weight": float(w),
            })

        data = {
            "metadata": {
                "sample_id": int(sample_id),
                "label": int(label),
                "num_nodes": len(nodes),
                "num_edges": len(json_edges),
            },
            "nodes": nodes,
            "edges": json_edges,
        }

        out_path = os.path.join(out_dir, f"pruned_{sample_id}.json")
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

    def _node_to_feature(self, n):
        denom_nodes = max(1, self.num_nodes - 1)

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
        head_idx = safe_float(raw_head_idx, 0.0) / max(1, self.num_heads)

       
        raw_ffn_idx = n.get("neuron_idx", n.get("ffn_idx", None))
        has_ffn_idx = 1.0 if raw_ffn_idx is not None else 0.0
        ffn_idx = safe_float(raw_ffn_idx, 0.0) / max(1, self.d_ff)

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

        feat = (
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

        return feat

    def get(self, idx):
        cached = self._data_cache[idx]
        if cached is not None:
            return cached

        path = self.graph_paths[idx]
        with open(path, "r") as f:
            obj = json.load(f)

        metadata = obj.get("metadata", {})
        label = metadata.get("label", 0)
        sample_id = metadata.get("sample_id", idx)
        y = torch.tensor([int(label)], dtype=torch.long)

        nodes = sorted(obj["nodes"], key=lambda n: int(n.get("id", 0)))
        edges = obj["edges"]

        if self.prune:
            heads = [n for n in nodes if n.get("node_type") == "head"]
            ffns = [n for n in nodes if n.get("node_type") == "ffn"]

            def prune_group(group, ratio):
                if not group:
                    return []
                sorted_group = sorted(
                    group,
                    key=lambda n: safe_float(n.get("score", 0.0)),
                    reverse=True,
                )
                keep_count = max(1, int(len(sorted_group) * ratio))
                return sorted_group[:keep_count]

            nodes = prune_group(heads, self.node_ratio) + prune_group(ffns, self.node_ratio)

        N = len(nodes)
        old_ids = [int(n["id"]) for n in nodes]
        id_to_new = {oid: i for i, oid in enumerate(old_ids)}

        src_list, dst_list, wts_list = [], [], []
        for e in edges:
            u = id_to_new.get(int(e["source"]))
            v = id_to_new.get(int(e["target"]))
            if u is not None and v is not None:
                src_list.append(u)
                dst_list.append(v)
                wts_list.append(safe_float(e.get("weight", 1.0)))

        if self.edge_prune and len(wts_list) > 0:
            w_tensor = torch.tensor(wts_list, dtype=torch.float)

            threshold = torch.quantile(w_tensor, 1.0 - self.edge_ratio).item()
            keep_indices = set()

            src_t = torch.tensor(src_list)
            dst_t = torch.tensor(dst_list)
            for i in range(N):
                mask = (src_t == i) | (dst_t == i)
                if mask.any():
                    idx_map = torch.where(mask)[0]
                    best_edge_local_idx = torch.argmax(w_tensor[idx_map])
                    keep_indices.add(idx_map[best_edge_local_idx].item())

            top_w_indices = torch.where(w_tensor >= threshold)[0].tolist()
            keep_indices.update(top_w_indices)

            final_idx = sorted(list(keep_indices))
            src_list = [src_list[i] for i in final_idx]
            dst_list = [dst_list[i] for i in final_idx]
            wts_list = [wts_list[i] for i in final_idx]

            if idx % 200 == 0:
                print(
                    f"  [Pruning] Sample {sample_id} | "
                    f"Threshold (Top {self.edge_ratio * 100:.1f}%): {threshold:.4f} | "
                    f"Kept {len(src_list)} edges",
                    flush=True,
                )

        if len(wts_list) > 0:
            w_tensor = torch.tensor(wts_list, dtype=torch.float)
            w_tensor = torch.log1p(w_tensor)
            max_w = w_tensor.max().item()
            if max_w > 0:
                w_tensor = w_tensor / max_w
            wts_list = w_tensor.tolist()
        else:
            wts_list = []

        nodes_out = []
        for i, n in enumerate(nodes):
            n_new = dict(n)
            n_new["orig_id"] = int(n_new.get("id", i))
            n_new["id"] = i
            nodes_out.append(n_new)

        if self.save_enabled and self.do_any_pruning:
            os.makedirs(self.derived_pruned_dir, exist_ok=True)
            self.save_json(
                sample_id,
                nodes_out,
                src_list,
                dst_list,
                wts_list,
                label,
                self.derived_pruned_dir,
            )

        X = []
        for n in nodes:
            feats = self._node_to_feature(n)
            X.append(feats)

        x = torch.tensor(X, dtype=torch.float)

        if len(src_list) > 0:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_weight = torch.tensor(wts_list, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.empty((0,), dtype=torch.float)

        if self.make_undirected and edge_index.numel() > 0:
            rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
            edge_index = torch.cat([edge_index, rev], dim=1)
            edge_weight = torch.cat([edge_weight, edge_weight], dim=0)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            sample_id=torch.tensor([int(sample_id)], dtype=torch.long),
        )
        self._data_cache[idx] = data
        return data

class GNNGraphClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 128,
        num_classes: int = 2,
        dropout: float = 0.2,
        model_type: str = "gcn",
        ggnn_steps: int = 3,
    ):
        super().__init__()
        self.model_type = model_type.lower()
        self.dropout = dropout

        if self.model_type == "gcn":
            self.conv1 = GCNConv(in_dim, hidden)
            self.conv2 = GCNConv(hidden, hidden)

        elif self.model_type == "gat":
            if hidden % 4 != 0:
                raise ValueError("For GAT, hidden must be divisible by 4.")
            self.conv1 = GATConv(in_dim, hidden // 4, heads=4)
            self.conv2 = GATConv(hidden, hidden // 4, heads=4)

        elif self.model_type == "gin":
            nn1 = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            nn2 = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.conv1 = GINConv(nn1)
            self.conv2 = GINConv(nn2)
        elif self.model_type == "ggnn":
            if in_dim > hidden:
                raise ValueError(
                    f"For GGNN, hidden ({hidden}) must be >= in_dim ({in_dim})."
                )
            self.conv1 = GatedGraphConv(hidden, num_layers=ggnn_steps)
            self.conv2 = None

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.lin = nn.Linear(hidden, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        if self.model_type == "ggnn":
            x = self.conv1(x, edge_index, edge_weight=edge_weight)
            x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            if self.model_type == "gcn":
                x = self.conv1(x, edge_index, edge_weight=edge_weight)
            else:
                x = self.conv1(x, edge_index)

            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.model_type == "gcn":
                x = self.conv2(x, edge_index, edge_weight=edge_weight)
            else:
                x = self.conv2(x, edge_index)

            x = F.relu(x)

        out = global_mean_pool(x, data.batch)
        out = self.lin(out)
        return out



def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total = 0
    total_correct = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        logits = model(batch)
        y = batch.y.view(-1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / max(1, total)
    acc = total_correct / max(1, total)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total = 0

    y_true, y_pred, sample_ids = [], [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        y = batch.y.view(-1)
        loss = criterion(logits, y)

        pred = logits.argmax(dim=-1)

        total_loss += loss.item() * y.size(0)
        total += y.size(0)

        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())

        sid = batch.sample_id.view(-1).cpu().numpy().tolist()
        sample_ids.extend(sid)

    avg_loss = total_loss / max(1, total)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = (y_true == y_pred).mean()

    prec_bin, rec_bin, f1_bin, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

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
        "ids": sample_ids,
        "true": y_true,
        "pred": y_pred,
    }



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", type=str, required=True, help="Path to graphs/train")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to graphs/test")

    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gin", "gat","ggnn"])

    parser.add_argument("--prune", action="store_true", help="Enable node pruning")
    parser.add_argument("--node_ratio", type=float, default=0.5, help="Ratio of nodes to keep")
    parser.add_argument("--edge_prune", action="store_true", help="Enable edge pruning")
    parser.add_argument("--edge_ratio", type=float, default=0.1, help="Top percentage of edges to keep")

    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_test", type=int, default=None)

    parser.add_argument("--val_ratio", type=float, default=0.15)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

   
    parser.add_argument("--run_name", type=str, default=None,
                        help="Base name for results files, e.g. wildjailbreak_gcn_t1")
    parser.add_argument("--ggnn_steps", type=int, default=3,
                        help="GGNN propagation rounds (shared-weight GRU steps)")
    parser.add_argument("--undirected", action="store_true",
                        help="Default is directed.")
    parser.add_argument(
        "--no_preload_train",
        action="store_true",
        help=(
            "Disable eager train/validation graph preloading. The dataset still "
            "uses a lazy in-memory cache, so graphs are cached on first access."
        ),
    )

    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(">>> Starting GNN training", flush=True)
    print(">>> Device:", device, flush=True)

    ds_params = {
        "prune": args.prune,
        "node_ratio": args.node_ratio,
        "edge_prune": args.edge_prune,
        "edge_ratio": args.edge_ratio,
        "make_undirected": args.undirected,
    }

    if args.prune or args.edge_prune:
        print(">>> Pruning requested. Materializing pruned train/test graphs...", flush=True)

        train_raw = JsonGraphDataset(args.train_dir, max_samples=args.max_train, **ds_params)
        test_raw = JsonGraphDataset(args.test_dir, max_samples=args.max_test, **ds_params)

        materialize_pruned(train_raw, desc="TRAIN")
        materialize_pruned(test_raw, desc="TEST")

        train_graph_dir = train_raw.derived_pruned_dir
        test_graph_dir = test_raw.derived_pruned_dir

        print(f">>> Using pruned train dir: {train_graph_dir}", flush=True)
        print(f">>> Using pruned test  dir: {test_graph_dir}", flush=True)

    else:
        print(">>> No pruning requested. Using original train/test directories.", flush=True)
        train_graph_dir = args.train_dir
        test_graph_dir = args.test_dir

    train_dataset = JsonGraphDataset(
        train_graph_dir,
        max_samples=args.max_train,
        prune=False,
        edge_prune=False,
        make_undirected=args.undirected,
    )

    test_dataset = JsonGraphDataset(
        test_graph_dir,
        max_samples=args.max_test,
        prune=False,
        edge_prune=False,
        make_undirected=args.undirected,
        num_layers=train_dataset.num_layers,
        pos_enc_dim=train_dataset.pos_enc_dim,
        num_heads=train_dataset.num_heads,  
        d_ff=train_dataset.d_ff,          
    )

    train_dataset.save_enabled = False
    test_dataset.save_enabled = False

    print(f">>> Train graphs total: {len(train_dataset)}", flush=True)
    print(f">>> Test graphs total:  {len(test_dataset)}", flush=True)
    print(f">>> Node feature dim:   {train_dataset.in_dim}", flush=True)

    sample = train_dataset.get(0)
    print(f">>> Example graph nodes: {sample.num_nodes}", flush=True)
    print(f">>> Example graph edges: {sample.num_edges}", flush=True)

    off = 3 + 2 + train_dataset.num_layers
    print(
        f">>> head_idx values on first graph: "
        f"{sample.x[:, off + 1].unique().tolist()[:8]}",
        flush=True,
    )
    print(
        f">>> ffn_idx values on first graph (sample):  "
        f"{sample.x[:, off + 3].unique().tolist()[:8]}",
        flush=True,
    )
    if not args.no_preload_train:
        train_dataset.preload(desc="TRAIN", report_every=100)
    else:
        print(
            ">>> Eager preloading disabled; lazy in-memory cache will fill "
            "during the first train/validation epoch.",
            flush=True,
        )

    all_indices = np.arange(len(train_dataset))
    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=args.val_ratio,
        random_state=args.seed,
        shuffle=True,
        stratify=train_dataset.labels,
    )

    print(f">>> Train split size: {len(train_idx)}", flush=True)
    print(f">>> Validation split size: {len(val_idx)}", flush=True)
    print(f">>> Train split label counts: {np.bincount(train_dataset.labels[train_idx])}", flush=True)
    print(f">>> Val split label counts:   {np.bincount(train_dataset.labels[val_idx])}", flush=True)
    print(f">>> Test label counts:        {np.bincount(test_dataset.labels)}", flush=True)

    train_subset = Subset(train_dataset, train_idx.tolist())
    val_subset = Subset(train_dataset, val_idx.tolist())
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = GNNGraphClassifier(
        in_dim=train_dataset.in_dim,
        hidden=args.hidden,
        dropout=args.dropout,
        model_type=args.model,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_metric = -1.0
    best_state = None
    best_epoch = -1

    for ep in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_metrics = evaluate(model, val_loader, criterion, device)

        if val_metrics["f1_macro"] > best_val_metric:
            best_val_metric = val_metrics["f1_macro"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep

        print(
            f"Ep {ep:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.3f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['acc']:.3f} | "
            f"Val Bin-F1: {val_metrics['f1_bin']:.3f} | "
            f"Val Macro-F1: {val_metrics['f1_macro']:.3f} | "
            f"Best Val Macro-F1: {best_val_metric:.3f} (Ep {best_epoch})",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val = evaluate(model, val_loader, criterion, device)
    final_test = evaluate(model, test_loader, criterion, device)

    print("\n=== Final Validation Metrics (Best Model) ===", flush=True)
    print(f"Accuracy:         {final_val['acc']:.4f}", flush=True)
    print(f"Binary Precision: {final_val['prec_bin']:.4f}", flush=True)
    print(f"Binary Recall:    {final_val['rec_bin']:.4f}", flush=True)
    print(f"Binary F1:        {final_val['f1_bin']:.4f}", flush=True)
    print(f"Macro Precision:  {final_val['prec_macro']:.4f}", flush=True)
    print(f"Macro Recall:     {final_val['rec_macro']:.4f}", flush=True)
    print(f"Macro F1:         {final_val['f1_macro']:.4f}", flush=True)

    print("\n=== Final Test Metrics (Best by Validation Macro-F1) ===", flush=True)
    print(f"Accuracy:         {final_test['acc']:.4f}", flush=True)
    print(f"Binary Precision: {final_test['prec_bin']:.4f}", flush=True)
    print(f"Binary Recall:    {final_test['rec_bin']:.4f}", flush=True)
    print(f"Binary F1:        {final_test['f1_bin']:.4f}", flush=True)
    print(f"Macro Precision:  {final_test['prec_macro']:.4f}", flush=True)
    print(f"Macro Recall:     {final_test['rec_macro']:.4f}", flush=True)
    print(f"Macro F1:         {final_test['f1_macro']:.4f}", flush=True)

    print("\nTest Confusion Matrix:", flush=True)
    print(final_test["cm"], flush=True)

    print("\nTest Classification Report:", flush=True)
    print(
        classification_report(
            final_test["true"],
            final_test["pred"],
            zero_division=0,
        ),
        flush=True,
    )

   

    os.makedirs("results", exist_ok=True)

    d_set = os.path.basename(os.path.normpath(args.train_dir))
    t_set = os.path.basename(os.path.normpath(args.test_dir))
    base = args.run_name or (
        f"{d_set}_to_{t_set}_{args.model}_prune{args.prune}_edge{args.edge_prune}"
    )

    df_details = pd.DataFrame({
        "sample_id": final_test["ids"],
        "true_label": final_test["true"],
        "predicted_label": final_test["pred"],
    })
    csv_path = os.path.join("results", f"{base}.csv")
    df_details.to_csv(csv_path, index=False)
    print(f">>> Detailed predictions saved to {csv_path}", flush=True)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "in_dim": train_dataset.in_dim,
        "num_layers": train_dataset.num_layers,
        "pos_enc_dim": train_dataset.pos_enc_dim,
        "num_heads": train_dataset.num_heads,  
        "d_ff": train_dataset.d_ff,           
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_metric,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "args": vars(args),
    }
    model_path = os.path.join("results", f"{base}.pt")
    torch.save(ckpt, model_path)
    print(f">>> Saved model checkpoint -> {model_path}", flush=True)


if __name__ == "__main__":
    main()