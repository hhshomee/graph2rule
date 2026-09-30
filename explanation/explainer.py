import argparse
import json
import os
import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, ModelConfig
from torch_geometric.explain.algorithm import GNNExplainer
from gnn import JsonGraphDataset, GNNGraphClassifier
from semantic import node_identity, identity_label, safe_float


class ExplainableGraphModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x, edge_index, batch, edge_weight=None):
        data = Data(x=x, edge_index=edge_index)
        data.batch = batch
        if edge_weight is not None:
            data.edge_weight = edge_weight
        return self.base_model(data)


def build_explainer(model, epochs, lr):
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs, lr=lr),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=ModelConfig(
            mode="multiclass_classification",
            task_level="graph",
            return_type="raw",
        ),
    )

def to_ranks(v):
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    if n <= 1:
        return np.ones(n)
    order = np.argsort(np.argsort(v))
    return order / (n - 1)


def coarsen(ident):
    ntype, layer, idx = ident
    if ntype == "ffn":
        return ("ffn", layer, -1)
    return ident

def explain_one_graph_multiseed(wrapper_model, data, device, seeds, epochs, lr):
    batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    edge_weight = getattr(data, "edge_weight", None)

    with torch.no_grad():
        logits = wrapper_model(x=data.x, edge_index=data.edge_index,
                               batch=batch, edge_weight=edge_weight)
        probs = logits.softmax(dim=-1).cpu().numpy().ravel()
        pred_class = int(logits.argmax(dim=-1).item())

    node_masks, edge_masks = [], []
    for s in range(seeds):
        torch.manual_seed(1000 + s)
        np.random.seed(1000 + s)
        explainer = build_explainer(wrapper_model, epochs=epochs, lr=lr)
        expl = explainer(x=data.x, edge_index=data.edge_index,
                         batch=batch, edge_weight=edge_weight)
        nm = expl.node_mask
        per_node = (nm.mean(dim=1) if nm.dim() == 2 else nm)
        node_masks.append(per_node.detach().cpu().numpy())
        edge_masks.append(expl.edge_mask.detach().cpu().numpy())

    node_masks = np.stack(node_masks)          # [S, N]
    edge_masks = np.stack(edge_masks)          # [S, E]

    # agreement: mean pairwise Pearson corr across seeds (node masks)
    if seeds > 1:
        cors = []
        for i in range(seeds):
            for j in range(i + 1, seeds):
                a, b = node_masks[i], node_masks[j]
                if a.std() < 1e-9 or b.std() < 1e-9:
                    continue
                cors.append(float(np.corrcoef(a, b)[0, 1]))
        agreement = float(np.mean(cors)) if cors else float("nan")
    else:
        agreement = float("nan")

    node_rank_avg = np.mean([to_ranks(m) for m in node_masks], axis=0)
    edge_rank_avg = np.mean([to_ranks(m) for m in edge_masks], axis=0)
    return node_rank_avg, edge_rank_avg, agreement, pred_class, probs

def extract_sample(node_ranks, edge_ranks, data, raw_obj,
                   top_k_edges, top_k_nodes, coarsen_fn=None):
    coarsen_fn = coarsen if coarsen_fn is None else coarsen_fn
    raw_nodes = sorted(raw_obj["nodes"], key=lambda n: int(n.get("id", 0)))

    head_ranks = {}
    ffn_by_layer = defaultdict(list)
    for i, n in enumerate(raw_nodes):
        ident = node_identity(n)
        if ident[0] == "head":
            head_ranks[ident] = float(node_ranks[i])
        elif ident[0] == "ffn":
            ffn_by_layer[ident[1]].append(float(node_ranks[i]))
    ffn_layer_ranks = {l: float(np.mean(v)) for l, v in ffn_by_layer.items()}

    # ---- top-k node records (coarsened) for circuits' node_summary ----
    if top_k_nodes < len(node_ranks):
        kept_nodes = set(np.argsort(-node_ranks)[:top_k_nodes].tolist())
    else:
        kept_nodes = set(range(len(node_ranks)))
    node_records = [(coarsen_fn(node_identity(raw_nodes[i])), float(node_ranks[i]))
                    for i in kept_nodes]

    # ---- top-k edge records (coarsened) ----
    ei = data.edge_index.cpu().numpy()
    id_to_new = {int(n["id"]): i for i, n in enumerate(raw_nodes)}
    etype_by_pair = {}
    for e in raw_obj["edges"]:
        s, t = int(e["source"]), int(e["target"])
        if s in id_to_new and t in id_to_new:
            etype_by_pair[(id_to_new[s], id_to_new[t])] = e.get("edge_type",
                                                                "unknown")
    if top_k_edges < len(edge_ranks):
        kept_edges = set(np.argsort(-edge_ranks)[:top_k_edges].tolist())
    else:
        kept_edges = set(range(len(edge_ranks)))

    edge_records = []
    for k in range(ei.shape[1]):
        if k not in kept_edges:
            continue
        u, v = int(ei[0, k]), int(ei[1, k])
        key = (coarsen_fn(node_identity(raw_nodes[u])),
               coarsen_fn(node_identity(raw_nodes[v])),
               etype_by_pair.get((u, v), "unknown"))
        edge_records.append((key, float(edge_ranks[k])))

    return dict(head_ranks=head_ranks, ffn_layer_ranks=ffn_layer_ranks,
                node_records=node_records, edge_records=edge_records)

def summarize_classwise(per_sample_data, num_classes=2, filter_mode="correct"):
    node_acc = {c: defaultdict(list) for c in range(num_classes)}
    edge_acc = {c: defaultdict(list) for c in range(num_classes)}
    class_counts = {c: 0 for c in range(num_classes)}

    for e in per_sample_data:
        c = e["pred_class"]
        if c not in class_counts:
            continue
        if filter_mode == "correct" and e["pred_class"] != e["true_label"]:
            continue
        if filter_mode == "incorrect" and e["pred_class"] == e["true_label"]:
            continue
        class_counts[c] += 1
        for ident, r in e["node_records"]:
            node_acc[c][ident].append(r)
        for ident, r in e["edge_records"]:
            edge_acc[c][ident].append(r)

    def unitize(acc, N_c):
        out = {}
        denom = max(1, N_c)
        for key, vals in acc.items():
            arr = np.array(vals)
            m = float(arr.mean())
            out[key] = dict(count=len(arr), freq=len(arr) / denom, mean=m,
                            weighted=m * len(arr) / denom,
                            std=float(arr.std()))
        return out

    node_summary = {c: unitize(node_acc[c], class_counts[c])
                    for c in range(num_classes)}
    edge_summary = {c: unitize(edge_acc[c], class_counts[c])
                    for c in range(num_classes)}

    def distinct(s0, s1):
        keys = set(s0) | set(s1)
        return {k: s1.get(k, {}).get("weighted", 0.0)
                   - s0.get(k, {}).get("weighted", 0.0)
                for k in keys}

    return {"node_summary": node_summary, "edge_summary": edge_summary,
            "node_distinct": distinct(node_summary[0], node_summary[1]),
            "edge_distinct": distinct(edge_summary[0], edge_summary[1]),
            "class_counts": class_counts, "filter_mode": filter_mode}

def perm_test(samples_scores, labels, n_perm=200, rng_seed=0):
    keys = sorted({k for d in samples_scores for k in d})
    S = len(samples_scores)
    M = np.full((S, len(keys)), np.nan)
    for si, d in enumerate(samples_scores):
        for ki, k in enumerate(keys):
            if k in d:
                M[si, ki] = d[k]
    labels = np.asarray(labels)

    def diffs(lab):
        with np.errstate(invalid="ignore"):
            m1 = np.nanmean(M[lab == 1], axis=0)
            m0 = np.nanmean(M[lab == 0], axis=0)
        return m1 - m0, m0, m1

    real, m0, m1 = diffs(labels)
    rng = np.random.default_rng(rng_seed)
    exceed = np.zeros(len(keys))
    for _ in range(n_perm):
        perm = rng.permutation(labels)
        d, _, _ = diffs(perm)
        exceed += (np.abs(np.nan_to_num(d)) >=
                   np.abs(np.nan_to_num(real))).astype(float)
    p = (exceed + 1) / (n_perm + 1)

    out = {}
    for ki, k in enumerate(keys):
        n0 = int(np.sum(~np.isnan(M[labels == 0, ki])))
        n1 = int(np.sum(~np.isnan(M[labels == 1, ki])))
        out[k] = (float(real[ki]), float(m0[ki]), float(m1[ki]),
                  n0, n1, float(p[ki]))
    return out

def write_stats_csv(path, stats, key_fmt, key_col):
    import csv
    rows = sorted(stats.items(), key=lambda kv: kv[1][5])  # by p-value
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([key_col, "mean_rank_c0", "mean_rank_c1",
                    "diff_c1_minus_c0", "n_c0", "n_c1", "p_value"])
        for k, (d, a0, a1, n0, n1, p) in rows:
            w.writerow([key_fmt(k), f"{a0:.4f}", f"{a1:.4f}", f"{d:+.4f}",
                        n0, n1, f"{p:.4f}"])


def write_verdict(path, agreement_vals, head_stats, ffn_stats,
                  pattern_lines, class_counts, alpha=0.05):
    ag = np.array([a for a in agreement_vals if not np.isnan(a)])
    ag_mean = float(ag.mean()) if len(ag) else float("nan")

    sig_heads = [(k, v) for k, v in head_stats.items() if v[5] < alpha]
    sig_heads.sort(key=lambda kv: kv[1][5])
    sig_ffn = [(k, v) for k, v in ffn_stats.items() if v[5] < alpha]
    sig_ffn.sort(key=lambda kv: kv[1][5])

    L = []
    L.append("=" * 60)
    L.append("VERDICT")
    L.append("=" * 60)
    L.append(f"Samples aggregated: class0={class_counts.get(0, 0)}  "
             f"class1={class_counts.get(1, 0)}")

    if np.isnan(ag_mean):
        L.append("Seed agreement:     n/a (single seed)")
    else:
        tag = ("good — averaging is safe" if ag_mean > 0.6 else
               "moderate — treat results with care" if ag_mean > 0.4 else
               "LOW — masks unstable; distinctions may be unreliable")
        L.append(f"Seed agreement:     {ag_mean:.2f}  ({tag})")

    L.append(f"\nSignificant heads (p<{alpha}): {len(sig_heads)}")
    for k, (d, a0, a1, n0, n1, p) in sig_heads[:10]:
        side = "class1" if d > 0 else "class0"
        L.append(f"  {identity_label(k):<10s} diff={d:+.3f} "
                 f"(favors {side})  p={p:.3f}")

    L.append(f"\nSignificant FFN layers (p<{alpha}): {len(sig_ffn)}")
    for k, (d, a0, a1, n0, n1, p) in sig_ffn:
        side = "class1" if d > 0 else "class0"
        L.append(f"  FFN-L{k}      diff={d:+.3f} (favors {side})  p={p:.3f}")

    if pattern_lines:
        L.append("\nTop class-distinctive motifs:")
        L.extend("  " + pl for pl in pattern_lines[:5])

    L.append("")
    if sig_heads or sig_ffn:
        L.append("Verdict: explanation signal DETECTED "
                 "(see CSVs; run ablation.py on the heads above).")
    else:
        L.append("Verdict: NO significant class distinction at current "
                 "settings. Do not over-interpret motif/circuit figures.")
    L.append("=" * 60)

    with open(path, "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--graph_dir", required=True)
    ap.add_argument("--out_dir", default=None,
                    help="default: explanations/<ckpt-stem>_<graphdir-stem>")
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=3,
                    help="explainer runs per graph (averaged)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--top_k_edges", type=int, default=40)
    ap.add_argument("--top_k_nodes", type=int, default=30)
    ap.add_argument("--aggregate", choices=["all", "correct", "incorrect"],
                    default="correct")
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--viz", type=int, default=0,
                    help="render N correctly-predicted graphs per class")
    ap.add_argument("--pattern_size", type=int, default=3, choices=[2, 3])
    ap.add_argument("--pattern_min_count", type=int, default=3)
    ap.add_argument("--no_circuits", action="store_true")
    ap.add_argument("--class_names", default="benign,harmful")
    ap.add_argument("--edge_keep_ratio", type=float, default=0.02)
    ap.add_argument("--min_edges", type=int, default=3)
    ap.add_argument("--min_nodes", type=int, default=4)
    ap.add_argument("--contrastive_keep_ratio", type=float, default=0.03)
    args = ap.parse_args()

    if args.out_dir is None:
        stem_c = os.path.splitext(os.path.basename(args.ckpt))[0]
        stem_g = os.path.basename(os.path.normpath(args.graph_dir))
        args.out_dir = os.path.join("explanations", f"{stem_c}__{stem_g}")
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    raw = torch.load(args.ckpt, map_location=device, weights_only=False)
    if all(isinstance(v, torch.Tensor) for v in raw.values()):
        state_dict = raw
        if "conv1.att_src" in state_dict or "conv1.att" in state_dict:
            model_type = "gat"
        elif "conv1.nn.0.weight" in state_dict:
            model_type = "gin"
        else:
            model_type = "gcn"
        w = (state_dict["conv1.nn.0.weight"] if model_type == "gin"
             else state_dict["conv1.lin.weight"])
        hidden, in_dim = w.shape
        ckpt = {"model_state_dict": state_dict,
                "args": {"model": model_type, "hidden": hidden,
                         "dropout": 0.2},
                "in_dim": in_dim, "num_layers": 6, "pos_enc_dim": 32}
        print(f"Bare state dict: inferred model={model_type} hidden={hidden}")
    else:
        ckpt = raw

    dataset = JsonGraphDataset(args.graph_dir, make_undirected=False,
                               max_samples=None,
                               num_layers=ckpt["num_layers"],
                               pos_enc_dim=ckpt["pos_enc_dim"])
    print(f"Dataset: {len(dataset)} graphs")

    base = GNNGraphClassifier(in_dim=ckpt["in_dim"],
                              hidden=ckpt["args"]["hidden"], num_classes=2,
                              dropout=ckpt["args"]["dropout"],
                              model_type=ckpt["args"]["model"]).to(device)
    base.load_state_dict(ckpt["model_state_dict"], strict=True)
    base.eval()
    model = ExplainableGraphModel(base).to(device).eval()

    start = max(0, args.start_idx)
    end = min(start + args.num_samples, len(dataset))
    print(f"Explaining graphs {start}..{end - 1}  seeds={args.seeds}  "
          f"epochs={args.epochs}")

    per_sample, agreements = [], []
    viz_done = {0: 0, 1: 0}

    for idx in range(start, end):
        data = dataset[idx].to(device)
        with open(dataset.graph_paths[idx]) as f:
            raw_obj = json.load(f)
        try:
            n_rank, e_rank, agree, pred, probs = explain_one_graph_multiseed(
                model, data, device, args.seeds, args.epochs, args.lr)
        except Exception as ex:
            print(f"  [{idx}] explain failed: {ex}")
            continue

        true_label = int(data.y.item())
        rec = extract_sample(n_rank, e_rank, data, raw_obj,
                             args.top_k_edges, args.top_k_nodes)
        rec.update(idx=idx, sample_id=int(data.sample_id.item()),
                   true_label=true_label, pred_class=pred,
                   probs=probs.tolist(), agreement=agree)
        per_sample.append(rec)
        agreements.append(agree)
        print(f"  [{idx - start + 1}/{end - start}] idx={idx} "
              f"true={true_label} pred={pred} agree={agree:.2f}"
              if not np.isnan(agree) else
              f"  [{idx - start + 1}/{end - start}] idx={idx} "
              f"true={true_label} pred={pred}")
    def keep(e):
        if args.aggregate == "correct":
            return e["pred_class"] == e["true_label"]
        if args.aggregate == "incorrect":
            return e["pred_class"] != e["true_label"]
        return True

    kept = [e for e in per_sample if keep(e)]
    labels = [e["pred_class"] for e in kept]
    head_stats = perm_test([e["head_ranks"] for e in kept], labels,
                           n_perm=args.n_perm)
    ffn_stats = perm_test([e["ffn_layer_ranks"] for e in kept], labels,
                          n_perm=args.n_perm)

    write_stats_csv(os.path.join(args.out_dir, "head_stats.csv"),
                    head_stats, identity_label, "head")
    write_stats_csv(os.path.join(args.out_dir, "ffn_layer_stats.csv"),
                    ffn_stats, lambda l: f"FFN-L{l}", "ffn_layer")
    with open(os.path.join(args.out_dir, "agreement.csv"), "w") as f:
        f.write("idx,true_label,pred_class,agreement\n")
        for e in per_sample:
            f.write(f"{e['idx']},{e['true_label']},{e['pred_class']},"
                    f"{e['agreement']:.4f}\n")
    with open(os.path.join(args.out_dir, "per_sample_log.json"), "w") as f:
        json.dump([{k: e[k] for k in
                    ("idx", "sample_id", "true_label", "pred_class", "probs")}
                   for e in per_sample], f, indent=2)
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump({**vars(args), "date": time.strftime("%Y-%m-%d %H:%M"),
                   "n_explained": len(per_sample)}, f, indent=2)
    summary = summarize_classwise(per_sample, filter_mode=args.aggregate)
    print(f"\nAll outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
