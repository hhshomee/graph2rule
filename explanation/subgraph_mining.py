import argparse
import collections
import csv
import itertools
import json
import os
from collections import defaultdict

import numpy as np

def build_granularity_fn(samples, granularity, topk_per_layer=16,
                         min_count=3, group_map=None, block_size=1024):
    if granularity == "topk":
        cnt = collections.Counter()
        for s in samples:
            seen = set()
            for src, dst, etype, score in s["edges"]:
                for n in (src, dst):
                    if n[0] == "ffn":
                        seen.add((int(n[1]), int(n[2])))
            for k in seen:
                cnt[k] += 1
        by_layer = collections.defaultdict(list)
        for (layer, idx), c in cnt.items():
            if c >= min_count:
                by_layer[layer].append((c, idx))
        keep = set()
        for layer, lst in by_layer.items():
            lst.sort(reverse=True)
            for c, idx in lst[:topk_per_layer]:
                keep.add((layer, idx))
        print(f"  [topk] keeping {len(keep)} individual FFN neurons "
              f"(>= {min_count} samples, <= {topk_per_layer}/layer); "
              f"remainder pooled per layer")

        def fn(ident):
            ntype, layer, idx = ident
            if ntype != "ffn":
                return f"H-L{layer}-{idx}"
            if (layer, idx) in keep:
                return f"F-L{layer}-{idx}"
            return f"F-L{layer}-rest"
        return fn

    if granularity == "block":
        b = max(1, block_size)

        def fn(ident):
            ntype, layer, idx = ident
            if ntype != "ffn":
                return f"H-L{layer}-{idx}"
            return f"F-L{layer}-b{idx // b}"
        return fn

    if granularity == "group":
        gm = group_map or {}
        n_unmapped = [0]

        def fn(ident):
            ntype, layer, idx = ident
            if ntype != "ffn":
                return f"H-L{layer}-{idx}"
            g = gm.get((layer, idx))
            if g is None:
                n_unmapped[0] += 1
                return f"F-L{layer}-gX"   
            return f"F-L{layer}-g{g}"
        fn._unmapped = n_unmapped
        return fn

    if granularity == "neuron":
        def fn(ident):
            ntype, layer, idx = ident
            return (f"F-L{layer}-{idx}" if ntype == "ffn"
                    else f"H-L{layer}-{idx}")
        return fn

    if granularity == "type":
        return lambda ident: f"{ident[0]}@L{ident[1]}"
    def fn(ident):
        ntype, layer, idx = ident
        return f"F-L{layer}" if ntype == "ffn" else f"H-L{layer}-{idx}"
    return fn


def label_identity(ident):
    ntype, layer, idx = ident
    tag = "H" if ntype == "head" else ("F" if ntype == "ffn" else "?")
    if ntype == "ffn" and idx == -1:
        return f"F-L{layer}"
    return f"{tag}-L{layer}-{idx}"


def label_type(ident):
    ntype, layer, _ = ident
    return f"{ntype}@L{layer}"


def layer_of(ident):
    return int(ident[1])


def cmd_dump(args):
    from explainer import (ExplainableGraphModel, explain_one_graph_multiseed,
                           extract_sample, coarsen)
    from gnn import JsonGraphDataset, GNNGraphClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw = torch.load(args.ckpt, map_location=device, weights_only=False)
    if all(isinstance(v, torch.Tensor) for v in raw.values()):
        sd = raw
        mt = ("gat" if ("conv1.att_src" in sd or "conv1.att" in sd)
              else "gin" if "conv1.nn.0.weight" in sd else "gcn")
        w = sd["conv1.nn.0.weight"] if mt == "gin" else sd["conv1.lin.weight"]
        hidden, in_dim = w.shape
        ckpt = {"model_state_dict": sd,
                "args": {"model": mt, "hidden": hidden, "dropout": 0.2},
                "in_dim": in_dim, "num_layers": 16, "pos_enc_dim": 32}
    else:
        ckpt = raw

    ds = JsonGraphDataset(args.graph_dir, make_undirected=False,
                          max_samples=None, num_layers=ckpt["num_layers"],
                          pos_enc_dim=ckpt["pos_enc_dim"])
    base = GNNGraphClassifier(in_dim=ckpt["in_dim"],
                              hidden=ckpt["args"]["hidden"], num_classes=2,
                              dropout=ckpt["args"]["dropout"],
                              model_type=ckpt["args"]["model"]).to(device)
    base.load_state_dict(ckpt["model_state_dict"], strict=True)
    base.eval()
    model = ExplainableGraphModel(base).to(device).eval()
    start = max(0, int(args.start_idx))
    if start >= len(ds):
        raise SystemExit(
            f"start_idx={start} is outside dataset of size {len(ds)}")
    end = min(start + int(args.num_samples), len(ds))
    n_chunk = end - start

    print(f"Dumping global graph indices [{start}, {end}) "
          f"({n_chunk} graphs) on device={device}", flush=True)

    out = []
    for local_i, idx in enumerate(range(start, end), 1):
        data = ds[idx].to(device)
        with open(ds.graph_paths[idx]) as f:
            raw_obj = json.load(f)
        try:
            n_rank, e_rank, agree, pred, probs = explain_one_graph_multiseed(
                model, data, device, args.seeds, args.epochs, args.lr)
        except Exception as ex:
            print(f"  [{idx}] failed: {ex}", flush=True)
            continue
        true = int(data.y.item())
        rec = extract_sample(n_rank, e_rank, data, raw_obj,
                             args.top_k_edges, args.top_k_nodes,
                             coarsen_fn=lambda ident: ident)
        out.append({
            "idx": idx,
            "true_label": true,
            "pred_class": pred,
            "edges": [[list(k[0]), list(k[1]), k[2], float(s)]
                      for k, s in rec["edge_records"]],
        })
        if local_i % 10 == 0 or local_i == n_chunk:
            print(f"  dumped {local_i}/{n_chunk} "
                  f"(global idx {idx})", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "graph_dir": args.graph_dir,
            "start_idx": start,
            "end_idx_exclusive": end,
            "requested": n_chunk,
            "completed": len(out),
            "samples": out,
        }, f)
    print(f"Wrote {len(out)}/{n_chunk} explanation subgraphs -> {args.out}")


def cmd_merge(args):
    import glob
    paths = sorted(glob.glob(os.path.join(args.in_dir, args.pattern)))
    if not paths:
        raise SystemExit(
            f"no chunk files matched {os.path.join(args.in_dir, args.pattern)}")

    merged = {}
    ckpt = graph_dir = None
    for path in paths:
        obj = json.load(open(path))
        if ckpt is None:
            ckpt = obj.get("ckpt")
            graph_dir = obj.get("graph_dir")
        else:
            if obj.get("ckpt") != ckpt:
                raise SystemExit(
                    f"checkpoint mismatch while merging: {path}")
            if obj.get("graph_dir") != graph_dir:
                raise SystemExit(
                    f"graph_dir mismatch while merging: {path}")

        for sample in obj.get("samples", []):
            idx = int(sample["idx"])
            if idx in merged:
                if merged[idx] != sample:
                    raise SystemExit(
                        f"conflicting duplicate sample idx={idx} in {path}")
                continue
            merged[idx] = sample

    idxs = sorted(merged)
    if not idxs:
        raise SystemExit("chunk files contained zero samples")

    expected_start = int(args.expected_start)
    expected_total = args.expected_total
    if expected_total is None:
        expected_end = idxs[-1] + 1
    else:
        expected_end = expected_start + int(expected_total)

    expected = set(range(expected_start, expected_end))
    missing = sorted(expected - set(idxs))
    extra = sorted(set(idxs) - expected)

    if missing and not args.allow_missing:
        preview = ", ".join(map(str, missing[:20]))
        more = "" if len(missing) <= 20 else f" ... (+{len(missing)-20})"
        raise SystemExit(
            f"merge incomplete: missing {len(missing)} expected samples: "
            f"{preview}{more}")
    if extra:
        print(f"[WARN] {len(extra)} samples lie outside expected range; "
              "keeping them because they were explicitly dumped.")

    samples = [merged[i] for i in idxs]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "ckpt": ckpt,
            "graph_dir": graph_dir,
            "merged_from": paths,
            "samples": samples,
        }, f)

    print(f"Merged {len(paths)} chunks -> {len(samples)} unique samples")
    if missing:
        print(f"[WARN] missing {len(missing)} expected samples")
    print(f"Wrote {args.out}")

def cmd_cluster(args):
    import glob
    from sklearn.cluster import KMeans

    paths = sorted(glob.glob(os.path.join(args.graph_dir, "*.json")))
    if args.max_graphs:
        paths = paths[:args.max_graphs]
    if not paths:
        raise SystemExit(f"no graphs in {args.graph_dir}")
    print(f"scanning {len(paths)} graphs for FFN neuron profiles...")

    FEATS = ["score", "max_val", "pos_com", "pos_ltr", "max_pos"]
    acc = collections.defaultdict(lambda: [np.zeros(len(FEATS)), 0])
    occ = collections.defaultdict(set)          # (layer,idx) -> sample indices

    for si, p in enumerate(paths):
        obj = json.load(open(p))
        for n in obj["nodes"]:
            if n.get("node_type") != "ffn":
                continue
            layer = int(round(float(n.get("layer_depth", 0))))
            idx = int(n.get("neuron_idx", n.get("ffn_idx", -1)))
            if idx < 0:
                continue
            v = np.array([float(n.get(f, 0.0) or 0.0) for f in FEATS])
            a = acc[(layer, idx)]
            a[0] += v
            a[1] += 1
            occ[(layer, idx)].add(si)
        if (si + 1) % 50 == 0:
            print(f"  {si + 1}/{len(paths)}", flush=True)

    by_layer = collections.defaultdict(list)
    for (layer, idx) in acc:
        by_layer[layer].append(idx)

    mapping, stats = {}, []
    for layer in sorted(by_layer):
        idxs = sorted(by_layer[layer])
        k = min(args.n_groups, len(idxs))
        if k < 2:
            mapping[f"L{layer}"] = {str(i): 0 for i in idxs}
            stats.append((layer, len(idxs), 1, [len(idxs)]))
            continue

        if args.method == "block":
            lab = np.array([i // args.block for i in idxs])
            uniq = {v: j for j, v in enumerate(sorted(set(lab.tolist())))}
            lab = np.array([uniq[v] for v in lab])
        else:
            if args.method == "profile":
                X = np.stack([acc[(layer, i)][0] / max(1, acc[(layer, i)][1])
                              for i in idxs])
            else:                                    # cooccur
                nS = len(paths)
                X = np.zeros((len(idxs), nS), dtype=np.float32)
                for r, i in enumerate(idxs):
                    for s in occ[(layer, i)]:
                        X[r, s] = 1.0
            mu, sd = X.mean(0), X.std(0) + 1e-9
            X = (X - mu) / sd
            lab = KMeans(n_clusters=k, n_init=10,
                         random_state=0).fit_predict(X)

        mapping[f"L{layer}"] = {str(i): int(g) for i, g in zip(idxs, lab)}
        sizes = sorted(collections.Counter(lab.tolist()).values(),
                       reverse=True)
        stats.append((layer, len(idxs), len(sizes), sizes))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"method": args.method, "n_groups": args.n_groups,
               "graph_dir": args.graph_dir, "map": mapping},
              open(args.out, "w"))

    print(f"\nmethod={args.method}  target groups/layer={args.n_groups}")
    print(f"{'layer':>6} {'neurons seen':>13} {'groups':>7}  group sizes")
    for layer, nidx, ng, sizes in stats:
        shown = ", ".join(str(s) for s in sizes[:8])
        more = "" if len(sizes) <= 8 else f", +{len(sizes) - 8} more"
        print(f"{layer:>6} {nidx:>13} {ng:>7}  {shown}{more}")
    print(f"\nwrote {args.out}")


def load_group_map(path):
    obj = json.load(open(path))
    m = obj["map"]
    return {(int(L[1:]), int(i)): int(g)
            for L, d in m.items() for i, g in d.items()}

def load_transactions(path, granularity, filter_mode, topk_per_layer=16,
                      min_count=3, group_map=None, block_size=1024):
    obj = json.load(open(path))
    lab_fn = build_granularity_fn(obj["samples"], granularity,
                                  topk_per_layer, min_count, group_map,
                                  block_size)

    if granularity in ("neuron", "topk", "group", "block"):
        has_neuron = any(int(n[2]) != -1
                         for s in obj["samples"][:20]
                         for e in s["edges"] for n in (e[0], e[1])
                         if n[0] == "ffn")
        if not has_neuron:
            print("  [ERROR] this dump has no neuron indices (all FFN idx are "
                  "-1): it was written before full-resolution dumping.\n"
                  "          Re-run `subgraph_mining.py dump` to use "
                  f"--granularity {granularity}.")
            raise SystemExit(1)

    txs, labels = [], []
    for s in obj["samples"]:
        if filter_mode == "correct" and s["pred_class"] != s["true_label"]:
            continue
        if filter_mode == "incorrect" and s["pred_class"] == s["true_label"]:
            continue
        edges = set()
        for src, dst, etype, score in s["edges"]:
            u = (src[0], int(src[1]), int(src[2]))
            v = (dst[0], int(dst[1]), int(dst[2]))
            lu, lv = lab_fn(u), lab_fn(v)
            if lu == lv:
                continue          # self-loop created by pooling; drop it
            edges.add((lu, lv, layer_of(u), layer_of(v)))
        txs.append(edges)
        labels.append(int(s["pred_class"]))
    return txs, np.array(labels)


def mine_frequent_edges(txs, min_count):
    cnt = defaultdict(int)
    for t in txs:
        for e in t:
            cnt[e] += 1
    return {e for e, c in cnt.items() if c >= min_count}


def subgraph_key(edges):
    return tuple(sorted((u, v) for u, v, _, _ in edges))


def support_sets(txs, freq_edges):
    occ = defaultdict(set)
    for i, t in enumerate(txs):
        for e in t:
            if e in freq_edges:
                occ[e].add(i)
    return occ


def grow(edges, occ, freq_edges, min_count, max_nodes):
    nodes = {u for u, v, _, _ in edges} | {v for u, v, _, _ in edges}
    cur_sup = None
    for e in edges:
        cur_sup = occ[e] if cur_sup is None else (cur_sup & occ[e])

    for e in freq_edges:
        if e in edges:
            continue
        u, v = e[0], e[1]
        if u not in nodes and v not in nodes:
            continue                      
        new_nodes = nodes | {u, v}
        if len(new_nodes) > max_nodes:
            continue
        sup = cur_sup & occ[e]
        if len(sup) < min_count:
            continue                     
        yield frozenset(edges | {e}), sup


def mine(txs, labels, min_support=0.10, max_nodes=6, max_edges=6,
         max_patterns=200000):
    n = len(txs)
    min_count = max(2, int(round(min_support * n)))
    freq_edges = mine_frequent_edges(txs, min_count)
    occ = support_sets(txs, freq_edges)
    print(f"  frequent single edges (support >= {min_count}/{n}): "
          f"{len(freq_edges)}")

    found = {}
    frontier = []
    for e in freq_edges:
        fs = frozenset({e})
        found[subgraph_key(fs)] = {"edges": fs, "sup": occ[e]}
        frontier.append(fs)

    size = 1
    while frontier and size < max_edges:
        size += 1
        nxt, seen = [], set()
        for edges in frontier:
            for new_edges, sup in grow(edges, occ, freq_edges, min_count,
                                       max_nodes):
                k = subgraph_key(new_edges)
                if k in seen:
                    continue
                seen.add(k)
                if k not in found:
                    found[k] = {"edges": new_edges, "sup": sup}
                    nxt.append(new_edges)
                if len(found) >= max_patterns:
                    print("  [warn] pattern cap reached; stopping growth")
                    return found
        frontier = nxt
        print(f"  size {size} edges: {len(nxt)} new subgraphs "
              f"(total {len(found)})")
    return found

def _pandas_append_shim():
    import pandas as pd
    if not hasattr(pd.DataFrame, "append"):
        def _append(self, other, ignore_index=False, **kw):
            other = pd.DataFrame([other]) if isinstance(other, dict) else other
            return pd.concat([self, other], ignore_index=ignore_index)
        pd.DataFrame.append = _append


def export_gspan(txs, path):
    lab2id, id2lab = {}, {}
    for t in txs:
        for u, v, _, _ in t:
            for x in (u, v):
                if x not in lab2id:
                    lab2id[x] = len(lab2id)
                    id2lab[lab2id[x]] = x

    out = []
    for gid, t in enumerate(txs):
        local = {}
        out.append(f"t # {gid}")
        for u, v, _, _ in sorted(t):
            for x in (u, v):
                if x not in local:
                    local[x] = len(local)
                    out.append(f"v {local[x]} {lab2id[x]}")
        for u, v, _, _ in sorted(t):
            out.append(f"e {local[u]} {local[v]} 0")
    out.append("t # -1")
    with open(path, "w") as f:
        f.write("\n".join(out))
    return lab2id, id2lab


def _decode_gspan_pattern(desc, id2lab):
    toks = desc.split()
    vlab, edges, i = {}, [], 0
    while i < len(toks):
        if toks[i] == "v":
            vlab[int(toks[i + 1])] = id2lab[int(toks[i + 2])]
            i += 3
        elif toks[i] == "e":
            edges.append((int(toks[i + 1]), int(toks[i + 2])))
            i += 4
        else:
            i += 1
    return {(vlab[a], vlab[b]) for a, b in edges if a in vlab and b in vlab}


def mine_gspan(txs, min_count, max_nodes, directed=False, verbose=False):
    import tempfile
    _pandas_append_shim()
    from gspan_mining.config import parser as gparser
    from gspan_mining.main import main as gspan_main

    path = os.path.join(tempfile.mkdtemp(), "db.data")
    lab2id, id2lab = export_gspan(txs, path)

    args = (f"-s {min_count} -d {directed} -l 1 -u {max_nodes} "
            f"-p False -w False -v False {path}")
    flags, _ = gparser.parse_known_args(args=args.split())
    gs = gspan_main(flags)

    tx_edges = [{(u, v) for u, v, _, _ in t} for t in txs]
    all_edges = set().union(*tx_edges) if tx_edges else set()

    def orient(pat):
        out = set()
        for a, b in pat:
            if (a, b) in all_edges:
                out.add((a, b))
            elif (b, a) in all_edges:
                out.add((b, a))
            else:
                return None            
        return out

    found = {}
    n_novertexedge = n_dup = n_lowsup = n_orient = 0
    for _, row in gs._report_df.iterrows():
        pat = _decode_gspan_pattern(row["description"], id2lab)
        if not pat:
            n_novertexedge += 1          # single-vertex pattern: no edges
            continue
        pat = orient(pat)
        if pat is None:
            n_orient += 1
            continue
        key = tuple(sorted(pat))
        if key in found:
            n_dup += 1
            continue
        sup = {i for i, te in enumerate(tx_edges) if pat <= te}
        if len(sup) < min_count:
            n_lowsup += 1
            continue
        found[key] = {"edges": pat, "sup": sup}
    if verbose:
        print(f"  gSpan reported {len(gs._report_df)} patterns "
              f"(mined undirected, re-oriented from data) -> {len(found)} kept "
              f"[dropped: {n_novertexedge} vertex-only, {n_dup} duplicate, "
              f"{n_orient} unorientable, {n_lowsup} below support]")
    return found


def score_patterns(found, labels, n_perm=0, rng_seed=0):
    n0 = int((labels == 0).sum())
    n1 = int((labels == 1).sum())
    rows = []
    idx0 = set(np.where(labels == 0)[0].tolist())
    idx1 = set(np.where(labels == 1)[0].tolist())

    for key, rec in found.items():
        sup = rec["sup"]
        c0 = len(sup & idx0)
        c1 = len(sup & idx1)
        f0 = c0 / max(1, n0)
        f1 = c1 / max(1, n1)
        rows.append({"key": key, "edges": rec["edges"], "sup": sup,
                     "n_nodes": len({u for u, v in key} | {v for u, v in key}),
                     "n_edges": len(key),
                     "c0": c0, "c1": c1, "f0": f0, "f1": f1,
                     "diff": f1 - f0, "p": None})

    if n_perm > 0:
        rng = np.random.default_rng(rng_seed)
        real = np.array([abs(r["diff"]) for r in rows])
        exceed = np.zeros(len(rows))
        supports = [r["sup"] for r in rows]
        for _ in range(n_perm):
            perm = rng.permutation(labels)
            p0 = set(np.where(perm == 0)[0].tolist())
            p1 = set(np.where(perm == 1)[0].tolist())
            d = np.array([abs(len(s & p1) / max(1, n1)
                              - len(s & p0) / max(1, n0)) for s in supports])
            exceed += (d >= real)
        for i, r in enumerate(rows):
            r["p"] = float((exceed[i] + 1) / (n_perm + 1))

    rows.sort(key=lambda r: (-abs(r["diff"]), -r["n_edges"]))
    return rows, n0, n1


def filter_closure(rows, mode="minimal"):
    if mode == "none":
        for r in rows:
            r["n_equiv"] = 1
        return rows

    groups = defaultdict(list)
    for r in rows:
        groups[frozenset(r["sup"])].append(r)

    kept = []
    for _, members in groups.items():
        members.sort(key=lambda r: (r["n_edges"], r["n_nodes"]))
        rep = members[0] if mode == "minimal" else members[-1]
        rep["n_equiv"] = len(members)
        kept.append(rep)

    kept.sort(key=lambda r: (-abs(r["diff"]), r["n_edges"]))
    return kept


def fmt_subgraph(key):
    return "  ".join(f"{u} -> {v}" for u, v in key)


def cmd_mine(args):
    gran = "layer" if args.granularity == "identity" else args.granularity
    gmap = None
    if gran == "group":
        if not args.groups:
            raise SystemExit("--granularity group needs --groups "
                             "(run the `cluster` command first)")
        gmap = load_group_map(args.groups)
        print(f"loaded group map: {len(gmap)} neurons assigned")
    if gran == "block":
        print(f"block granularity: {args.block_size} neurons/block "
              f"(~{8192 // max(1, args.block_size)} groups per layer at "
              f"d_ff=8192)")
    txs, labels = load_transactions(args.infile, gran, args.aggregate,
                                    args.topk_per_layer,
                                    args.min_neuron_count, gmap,
                                    args.block_size)
    print(f"transactions: {len(txs)}  "
          f"(class0={int((labels==0).sum())}, class1={int((labels==1).sum())})")
    print(f"granularity: {gran}")

    if args.engine == "gspan":
        min_count = max(2, int(round(args.min_support * len(txs))))
        print(f"engine: gSpan (min_count={min_count}/{len(txs)})")
        found = mine_gspan(txs, min_count, args.max_nodes, verbose=True)
    else:
        found = mine(txs, labels, min_support=args.min_support,
                     max_nodes=args.max_nodes, max_edges=args.max_edges)
    rows, n0, n1 = score_patterns(found, labels, n_perm=args.n_perm)
    n_raw = len(rows)
    rows = filter_closure(rows, mode=args.closure)
    print(f"  closure filter ({args.closure}): {n_raw} -> {len(rows)} patterns")

    out_dir = args.out_dir or os.path.dirname(args.infile) or "."
    os.makedirs(out_dir, exist_ok=True)
    stem = (f"mined_subgraphs_{gran}{args.block_size}"
            if gran == "block" else f"mined_subgraphs_{gran}")
    if args.engine == "gspan":
        stem += "_gspan"

    csv_path = os.path.join(out_dir, stem + ".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subgraph", "n_nodes", "n_edges", "n_equiv_collapsed",
                    "count_c0", "count_c1", "support_c0", "support_c1",
                    "diff_c1_minus_c0", "p_value"])
        for r in rows:
            w.writerow([fmt_subgraph(r["key"]), r["n_nodes"], r["n_edges"],
                        r.get("n_equiv", 1),
                        r["c0"], r["c1"], f"{r['f0']:.4f}", f"{r['f1']:.4f}",
                        f"{r['diff']:+.4f}",
                        "" if r["p"] is None else f"{r['p']:.4f}"])

    # nodes of the single most discriminative subgraph per class -> ablation
    top1 = next((r for r in rows if r["diff"] > 0), None)
    top0 = next((r for r in rows if r["diff"] < 0), None)
    payload = {}
    for cls, r in (("class1", top1), ("class0", top0)):
        if r is None:
            continue
        nodes = sorted({u for u, v in r["key"]} | {v for u, v in r["key"]})
        payload[cls] = {"nodes": nodes, "subgraph": fmt_subgraph(r["key"]),
                        "support_c0": r["f0"], "support_c1": r["f1"],
                        "diff": r["diff"], "p_value": r["p"]}
    json_path = os.path.join(out_dir, stem + "_top.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    lines = ["=" * 66,
             f"MINED DISCRIMINATIVE SUBGRAPHS ({gran} labels)",
             "=" * 66,
             f"transactions: {len(txs)}  class0={n0}  class1={n1}",
             f"min_support={args.min_support}  max_nodes={args.max_nodes}  "
             f"max_edges={args.max_edges}  closure={args.closure}",
             f"distinct subgraphs: {len(rows)} (from {n_raw} before closure "
             f"filtering)", ""]
    for title, sign in (("Most CLASS-1 discriminative", 1),
                        ("Most CLASS-0 discriminative", -1)):
        lines.append(title + ":")
        shown = 0
        for r in rows:
            if sign * r["diff"] <= 0:
                continue
            pstr = "" if r["p"] is None else f"  p={r['p']:.3f}"
            eq = r.get("n_equiv", 1)
            eqs = "" if eq <= 1 else f"  (+{eq - 1} supersets, same support)"
            lines.append(f"  c0={r['f0']:5.1%}  c1={r['f1']:5.1%}  "
                         f"diff={r['diff']:+.3f}{pstr}{eqs}")
            lines.append(f"     [{r['n_nodes']}n/{r['n_edges']}e] "
                         f"{fmt_subgraph(r['key'])}")
            shown += 1
            if shown >= args.top_k:
                break
        if shown == 0:
            lines.append("  (none)")
        lines.append("")
    lines.append("=" * 66)
    report = "\n".join(lines)
    with open(os.path.join(out_dir, stem + "_report.txt"), "w") as f:
        f.write(report)
    print(report)
    print(f"\nCSV:  {csv_path}\nTop:  {json_path}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump", help="record explanation subgraphs")
    d.add_argument("--ckpt", required=True)
    d.add_argument("--graph_dir", required=True)
    d.add_argument("--out", default=None)
    d.add_argument("--start_idx", type=int, default=0,
                   help="global dataset index at which this dump chunk starts")
    d.add_argument("--num_samples", type=int, default=100000,
                   help="maximum number of graphs in this dump chunk")
    d.add_argument("--seeds", type=int, default=2)
    d.add_argument("--epochs", type=int, default=100)
    d.add_argument("--lr", type=float, default=0.01)
    d.add_argument("--top_k_edges", type=int, default=40)
    d.add_argument("--top_k_nodes", type=int, default=30)
    d.set_defaults(func=cmd_dump)

    mg = sub.add_parser("merge", help="merge chunked explanation dumps")
    mg.add_argument("--in_dir", required=True)
    mg.add_argument("--pattern", default="chunk_*.json")
    mg.add_argument("--out", required=True)
    mg.add_argument("--expected_start", type=int, default=0)
    mg.add_argument("--expected_total", type=int, default=None,
                    help="expected number of global samples; if supplied, "
                         "merge fails on missing indices unless --allow_missing")
    mg.add_argument("--allow_missing", action="store_true")
    mg.set_defaults(func=cmd_merge)

    c = sub.add_parser("cluster", help="group FFN neurons into K per layer")
    c.add_argument("--graph_dir", required=True,
                   help="the graphs the dump was made from")
    c.add_argument("--out", default="ffn_groups.json")
    c.add_argument("--n_groups", type=int, default=8,
                   help="target groups per layer")
    c.add_argument("--method", choices=["profile", "cooccur", "block"],
                   default="profile")
    c.add_argument("--block", type=int, default=1024,
                   help="for --method block: neurons per block")
    c.add_argument("--max_graphs", type=int, default=None)
    c.set_defaults(func=cmd_cluster)

    m = sub.add_parser("mine", help="mine discriminative subgraphs")
    m.add_argument("--in", dest="infile", required=True)
    m.add_argument("--granularity",
                   choices=["neuron", "block", "group", "topk", "layer",
                            "type", "identity"],
                   default="layer",
                   help="neuron: every FFN neuron kept | topk: frequent "
                        "neurons kept, rest pooled per layer (MIDDLE GROUND) "
                        "| layer: all FFN of a layer merged (original) | "
                        "type: also drops head indices "
                        "('identity' is an alias for 'layer')")
    m.add_argument("--topk_per_layer", type=int, default=16,
                   help="for --granularity topk: individual neurons kept "
                        "per layer")
    m.add_argument("--block_size", type=int, default=1024,
                   help="for --granularity block: neurons per block. "
                        "d_ff / block_size = groups per layer "
                        "(8192/1024 = 8). Deterministic, no corpus pass, "
                        "no dependence on the sample set.")
    m.add_argument("--groups", default=None,
                   help="neuron->group map from the `cluster` command; "
                        "required for --granularity group")
    m.add_argument("--min_neuron_count", type=int, default=3,
                   help="for --granularity topk: a neuron must appear in at "
                        "least this many samples to keep its own identity")
    m.add_argument("--aggregate", choices=["all", "correct", "incorrect"],
                   default="correct")
    m.add_argument("--min_support", type=float, default=0.10)
    m.add_argument("--max_nodes", type=int, default=6)
    m.add_argument("--max_edges", type=int, default=6)
    m.add_argument("--n_perm", type=int, default=200)
    m.add_argument("--engine", choices=["builtin", "gspan"],
                   default="builtin",
                   help="builtin: Apriori-style level-wise miner (this file) | "
                        "gspan: enumerate with the gSpan algorithm "
                        "(Yan & Han 2002) via the gspan-mining package")
    m.add_argument("--closure", choices=["minimal", "maximal", "none"],
                   default="minimal",
                   help="collapse patterns sharing an identical support set")
    m.add_argument("--top_k", type=int, default=10)
    m.add_argument("--out_dir", default=None)
    m.set_defaults(func=cmd_mine)

    args = ap.parse_args()
    if args.cmd == "dump" and args.out is None:
        stem_c = os.path.splitext(os.path.basename(args.ckpt))[0]
        stem_g = os.path.basename(os.path.normpath(args.graph_dir))
        args.out = os.path.join("explanations", f"{stem_c}__{stem_g}",
                                "explanation_subgraphs.json")
    args.func(args)


if __name__ == "__main__":
    main()
