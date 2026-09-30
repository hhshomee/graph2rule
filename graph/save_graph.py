import argparse
import json
import math
import os
import pickle
from typing import Dict, List, Tuple

import networkx as nx
import math
import torch
import torch.nn.functional as F
import torch.nn.functional as F

def get_sinusoidal_encoding(pos_norm: float, d_model: int = 32):
    """
    Generate a sinusoidal positional encoding vector
    from a normalized position pos_norm in [0, 1].
    """
    pe = torch.zeros(d_model)
    position = torch.tensor([pos_norm * 512.0])  # 512 = assumed max length

    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )

    pe[0::2] = torch.sin(position * div_term)
    pe[1::2] = torch.cos(position * div_term)
    return pe.tolist()
def get_positional_stats(activations: torch.Tensor) -> dict:
    T = activations.shape[0]
    if T == 0:
        return {"mean": 0, "max": 0, "com": 0, "ltr": 0, "max_pos": 0}
    
    a = activations.abs() 
    if a.ndim > 1:
        a = a.mean(dim=-1)   
    sum_a = a.sum().item() + 1e-9
    max_pos = a.argmax().item()
    
    t_range = torch.arange(T, device=a.device).float()
    com = (a * t_range).sum().item() / sum_a
    norm_com = com / T
    
    cutoff = int(0.7 * T)
    ltr = a[cutoff:].sum().item() / sum_a
    
    return {
        "score": a.mean().item(),
        "max_val": a.max().item(),
        "pos_com": norm_com,
        "pos_ltr": ltr,
        "max_pos": max_pos  
    }

def attention_concentration(attn: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:

    attn = attn.float()                            
    if attn.shape[-1] <= 2:
        p = attn.clamp(min=eps)
    else:
        p = attn[..., 1:].clamp(min=eps)           # drop sink column [H, T, T-1]
        p = p / p.sum(dim=-1, keepdim=True)        # renormalize rows
    ent = -(p * p.log()).sum(dim=-1)               # [H, T]
    K = p.shape[-1]
    ent_norm = ent / math.log(max(K, 2))           # [H, T]
    if ent_norm.shape[-1] > 2:
        ent_norm = ent_norm[..., 1:]
    score = 1.0 - ent_norm.mean(dim=-1)            # [H]
    return score


def topk_indices(x: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, x.numel())
    if k <= 0:
        return torch.empty((0,), dtype=torch.long)
    return torch.topk(x, k=k, largest=True).indices

def calc_pretrained_weight(src_node, dst_node, weight_dict):
    l_src, l_dst = src_node["layer_depth"], dst_node["layer_depth"]
    
    # CASE 1: Head -> Head (L to L+1) 
    if src_node["node_type"] == "head" and dst_node["node_type"] == "head":
        W_o = weight_dict[l_src]["heads"][src_node["head_idx"]]["W_o"]
        W_q = weight_dict[l_dst]["heads"][dst_node["head_idx"]]["W_q"]
        return torch.norm(torch.matmul(W_q, W_o), p='fro').item()

    # CASE 2: Head -> FFN (Within Layer L)
    if src_node["node_type"] == "head" and dst_node["node_type"] == "ffn":
        W_o = weight_dict[l_src]["heads"][src_node["head_idx"]]["W_o"]
        W_wi_k = weight_dict[l_dst]["ffn_wi"][dst_node["neuron_idx"]]
        return torch.norm(torch.matmul(W_wi_k, W_o)).item()

    # CASE 3: FFN Neuron (Layer L) -> Head (Layer L+1)
    if src_node["node_type"] == "ffn" and dst_node["node_type"] == "head":
        W_wo_k = weight_dict[l_src]["ffn_wo"][:, src_node["neuron_idx"]]
        W_q = weight_dict[l_dst]["heads"][dst_node["head_idx"]]["W_q"]
        return torch.norm(torch.matmul(W_q, W_wo_k)).item()

    # CASE 4: FFN Neuron (Layer L) -> FFN Neuron (Layer L+1)
    if src_node["node_type"] == "ffn" and dst_node["node_type"] == "ffn":
        W_wo_k = weight_dict[l_src]["ffn_wo"][:, src_node["neuron_idx"]]
        W_wi_next = weight_dict[l_dst]["ffn_wi"][dst_node["neuron_idx"]]
        return torch.abs(torch.dot(W_wi_next, W_wo_k)).item()
        
    return 0.05

def build_graph_from_sample(
    payload: Dict,
    weight_dict: Dict = None,
    topk_heads: int = 4,
    topk_ffn: int = 64,
    cross_layer_edges: bool = True,
    within_layer_head_to_ffn: bool = True,
    edge_weight_mode: str = "pretrained", 
) -> nx.DiGraph:
    attn_list: List[torch.Tensor] = payload["encoder_attentions"]
    ffn_dict: Dict[int, torch.Tensor] = payload["ffn_wi_out"]
    mask = payload.get("attention_mask", None)
    if mask is not None:
        keep = mask.bool().cpu()
        if int(keep.sum()) < keep.numel():
            attn_list = [a[:, keep][:, :, keep] for a in attn_list]
            ffn_dict = {l: t[keep] for l, t in ffn_dict.items()}

    L = len(attn_list) 
    G = nx.DiGraph()

    global_node_id = 0
    mapping = {}  
    layer_nodes = {l: [] for l in range(L)}  
    for l in range(L):
        attn = attn_list[l]
        if attn.is_cuda:
            attn = attn.cpu()
        head_scores = attention_concentration(attn)  # [H]
        keep_heads = topk_indices(head_scores, topk_heads)

        for h in keep_heads.tolist():
            nid = global_node_id
            mapping[(l, "head", int(h))] = nid  
            A = attn[h].float()
            if A.shape[-1] > 2:
                A = A[1:, 1:]                               
                A = A / (A.sum(dim=-1, keepdim=True) + 1e-9)
            head_acts = A.mean(dim=0)
            stats = get_positional_stats(head_acts)
            stats["score"] = float(head_scores[h])

            G.add_node(
                nid,
                node_type="head",
                layer_depth=int(l),
                head_idx=int(h),
                pos_enc=get_sinusoidal_encoding(stats['pos_com'], d_model=32),
                **stats
            )
            layer_nodes[l].append(nid)
            global_node_id += 1
             
    for l in range(L):
        if l not in ffn_dict:
            continue
        ffn = ffn_dict[l]
        if ffn.is_cuda:
            ffn = ffn.cpu()

        neuron_scores = ffn.abs().mean(dim=0)

        keep_neurons = topk_indices(neuron_scores, topk_ffn)
        for k in keep_neurons.tolist():
            nid = global_node_id
            mapping[(l, "ffn", int(k))] = nid
            neuron_acts = ffn[:, k] # [T]
            pos_stats = get_positional_stats(neuron_acts)
            pos_enc = get_sinusoidal_encoding(pos_stats['pos_com'], d_model=32)
            
            G.add_node(
                nid,
                node_type="ffn",
                layer_depth=int(l),
                neuron_idx=int(k),
                phys_x=int(l),
                phys_y=int(k),
                pos_enc=pos_enc,
                **pos_stats 
            )
            layer_nodes[l].append(nid)
            global_node_id += 1
    all_scores = [float(d.get("score", 0.0)) for _, d in G.nodes(data=True)]
    score_max = max(all_scores) + 1e-8 

    def get_final_weight(u, v):
        if edge_weight_mode == "pretrained" and weight_dict:
            return calc_pretrained_weight(G.nodes[u], G.nodes[v], weight_dict)
        s, t = G.nodes[u]["score"], G.nodes[v]["score"]
        return s * t if edge_weight_mode == "product" else s + t
    if within_layer_head_to_ffn:
        for l in range(L):
            heads = [nid for nid in layer_nodes[l] if G.nodes[nid]["node_type"] == "head"]
            ffns  = [nid for nid in layer_nodes[l] if G.nodes[nid]["node_type"] == "ffn"]
            for u in heads:
                for v in ffns:
                    G.add_edge(u, v, edge_type="head_to_ffn", weight=get_final_weight(u, v))
    if cross_layer_edges:
        for l in range(L - 1):
            for u in layer_nodes[l]:
                src_type = G.nodes[u]["node_type"]
                for v in layer_nodes[l + 1]:
                    dst_type = G.nodes[v]["node_type"]
                    if src_type == "head" and dst_type == "head":
                        G.add_edge(
                            u,
                            v,
                            edge_type="head_to_head_next",
                            weight=get_final_weight(u, v),
                        )
                    elif src_type == "ffn" and dst_type == "head":
                        G.add_edge(
                            u,
                            v,
                            edge_type="ffn_to_head_next",
                            weight=get_final_weight(u, v),
                        )
                    elif src_type == "ffn" and dst_type == "ffn":
                        G.add_edge(
                            u,
                            v,
                            edge_type="ffn_to_ffn_next",
                            weight=get_final_weight(u, v),
                        )

    return G


def save_nx_graph(G: nx.DiGraph, out_prefix: str,save_json=True):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    layer_counts = {}
    for _, d in G.nodes(data=True):
        l = d["layer_depth"]
        layer_counts[l] = layer_counts.get(l, 0) + 1

    data = {
        "metadata": {
            "label": G.graph.get("label"), 
            "sample_id": G.graph.get("sample_id"),
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "density": nx.density(G),
            "is_directed": G.is_directed(),
            "nodes_per_layer": layer_counts,
            "out_prefix": out_prefix,
        },
        "nodes": [{"id": n, **G.nodes[n]} for n in G.nodes],
        "edges": [{"source": u, "target": v, **G[u][v]} for u, v in G.edges],
    }
    if save_json:
        with open(out_prefix + ".json", "w") as f:
            json.dump(data, f, separators=(",", ":"))  


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensors_dir", type=str, required=True, help="Dir with per-sample .pt tensors")
    parser.add_argument("--out_dir", type=str, default="graphs", help="Output dir for graphs")
    parser.add_argument("--max_samples", type=int, default=None, help="Process only first N tensor files")
    parser.add_argument("--topk_heads", type=int, default=4, help="Top-K heads per layer")
    parser.add_argument("--topk_ffn", type=int, default=64, help="Top-K FFN neurons per layer")
    parser.add_argument("--no_cross_layer", action="store_true", help="Disable cross-layer edges")
    parser.add_argument("--no_head_to_ffn", action="store_true", help="Disable within-layer head->ffn edges")
    parser.add_argument("--edge_weight_mode", type=str, default="pretrained")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pt_files = sorted([f for f in os.listdir(args.tensors_dir) if f.endswith(".pt")])
    if args.max_samples is not None:
        pt_files = pt_files[:args.max_samples]

    for fname in pt_files:
        path = os.path.join(args.tensors_dir, fname)
        payload = torch.load(path, map_location="cpu")

        sid = payload.get("id", os.path.splitext(fname)[0])

        G = build_graph_from_sample(
            payload,
            topk_heads=args.topk_heads,
            topk_ffn=args.topk_ffn,
            cross_layer_edges=not args.no_cross_layer,
            within_layer_head_to_ffn=not args.no_head_to_ffn,
            edge_weight_mode=args.edge_weight_mode,
        )

        out_prefix = os.path.join(args.out_dir, f"graph_{sid}")
        save_nx_graph(G, out_prefix)

        print(f"[saved] {out_prefix}.json  nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    print(f"Done. Graphs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()