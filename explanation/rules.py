import argparse
import csv
import json
import os
import re
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

def block_label(ident, block_size):
    ntype, layer, idx = ident[0], int(ident[1]), int(ident[2])
    if ntype != "ffn":
        return f"H-L{layer}-{idx}"
    return f"F-L{layer}-b{idx // block_size}"


def layer_label(ident):
    ntype, layer, idx = ident[0], int(ident[1]), int(ident[2])
    return f"F-L{layer}" if ntype == "ffn" else f"H-L{layer}-{idx}"


def load_dump(path, block_size, granularity):
    obj = json.load(open(path))
    lab = ((lambda i: block_label(i, block_size)) if granularity == "block"
           else layer_label)
    out = []
    for s in obj["samples"]:
        edges = set()
        for src, dst, etype, score in s["edges"]:
            u, v = lab(tuple(src)), lab(tuple(dst))
            if u != v:
                edges.add((u, v))
        out.append({"edges": edges, "pred": int(s["pred_class"]),
                    "true": int(s["true_label"])})
    return out


# -------------------- concepts --------------------

def parse_subgraph(text):
    edges = set()
    for part in re.split(r"\s{2,}", text.strip()):
        if "->" not in part:
            continue
        a, b = [x.strip() for x in part.split("->")]
        if a and b:
            edges.add((a, b))
    return frozenset(edges)


def load_concepts(csv_path, top_k, min_edges=1):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if int(r["n_edges"]) < min_edges:
                continue
            pat = parse_subgraph(r["subgraph"])
            if pat:
                rows.append((abs(float(r["diff_c1_minus_c0"])),
                             r["subgraph"].strip(), pat))
    rows.sort(key=lambda t: -t[0])

    seen, out = set(), []
    for _, name, pat in rows:
        if pat in seen:
            continue
        seen.add(pat)
        out.append((name, pat))
        if len(out) >= top_k:
            break
    return out


def featurize(samples, concepts):
    X = np.zeros((len(samples), len(concepts)), dtype=np.int8)
    for i, s in enumerate(samples):
        for j, (_, pat) in enumerate(concepts):
            if pat <= s["edges"]:
                X[i, j] = 1
    return X


# -------------------- rule --------------------

def simplify(clauses):
    cur = [frozenset(c) for c in clauses]
    changed = True
    while changed:
        changed = False
        out, used = [], set()
        for i in range(len(cur)):
            if i in used:
                continue
            merged = False
            for j in range(i + 1, len(cur)):
                if j in used:
                    continue
                a, b = cur[i], cur[j]
                diff = a ^ b                      # symmetric difference
                if len(diff) != 2:
                    continue
                x, y = sorted(diff, key=len)
                if y == f"NOT {x}" or x == f"NOT {y}":
                    out.append(a & b)             # drop the irrelevant literal
                    used.update({i, j})
                    merged = changed = True
                    break
            if not merged and i not in used:
                out.append(cur[i])
                used.add(i)
        cur = out
    cur = sorted(set(cur), key=len)
    keep = []
    for c in cur:
        if not any(k <= c for k in keep):
            keep.append(c)
    return keep


def fmt_clauses(clauses):
    if not clauses:
        return "(never predicts this class)"
    return "\n     OR ".join(
        "(" + " AND ".join(sorted(c, key=lambda s: (s.startswith("NOT"), s)))
        + ")" for c in clauses)


def tree_to_rule(tree, names, target=1):
    t = tree.tree_
    leaves = []

    def walk(node, conds):
        if t.children_left[node] == -1:
            frac = t.value[node][0]               
            n = int(t.n_node_samples[node])       
            if int(np.argmax(frac)) == target:
                leaves.append((list(conds), n, float(frac[target])))
            return
        name = names[t.feature[node]]
        walk(t.children_left[node], conds + [f"NOT {name}"])   # feature == 0
        walk(t.children_right[node], conds + [name])           # feature == 1

    walk(0, [])
    if not leaves:
        return "(never predicts this class)", "(never predicts this class)", []
    raw = fmt_clauses([frozenset(c) for c, _, _ in leaves])
    simple = fmt_clauses(simplify([c for c, _, _ in leaves]))
    return raw, simple, leaves


def fidelity(pred_rule, pred_gnn):
    return float((np.asarray(pred_rule) == np.asarray(pred_gnn)).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_mined", required=True,
                    help="mined_subgraphs_*.csv from the TRAIN dump")
    ap.add_argument("--train_dump", required=True)
    ap.add_argument("--test_dump", required=True)
    ap.add_argument("--granularity", choices=["block", "layer"],
                    default="block")
    ap.add_argument("--block_size", type=int, default=1024)
    ap.add_argument("--top_k", type=int, default=15,
                    help="how many mined subgraphs become features")
    ap.add_argument("--depth", type=int, default=3,
                    help="max decision-tree depth (rule complexity)")
    ap.add_argument("--min_edges", type=int, default=1)
    ap.add_argument("--n_random", type=int, default=20,
                    help="random-concept baseline repetitions")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    train = load_dump(args.train_dump, args.block_size, args.granularity)
    test = load_dump(args.test_dump, args.block_size, args.granularity)
    concepts = load_concepts(args.train_mined, args.top_k, args.min_edges)
    if not concepts:
        raise SystemExit("no concepts parsed from " + args.train_mined)

    names = [f"C{j}" for j in range(len(concepts))]
    Xtr, Xte = featurize(train, concepts), featurize(test, concepts)
    ytr = np.array([s["pred"] for s in train])      # target = GNN's prediction
    yte = np.array([s["pred"] for s in test])

    clf = DecisionTreeClassifier(max_depth=args.depth, random_state=0)
    clf.fit(Xtr, ytr)
    fid_tr = fidelity(clf.predict(Xtr), ytr)
    fid_te = fidelity(clf.predict(Xte), yte)

    # baselines
    maj = int(np.bincount(ytr).argmax())
    fid_maj = fidelity(np.full(len(yte), maj), yte)

    rng = np.random.default_rng(0)
    all_rows = load_concepts(args.train_mined, 10 ** 6, args.min_edges)
    rand_scores = []
    if len(all_rows) > len(concepts):
        for _ in range(args.n_random):
            pick = rng.choice(len(all_rows), size=len(concepts), replace=False)
            rc = [all_rows[i] for i in pick]
            c = DecisionTreeClassifier(max_depth=args.depth, random_state=0)
            c.fit(featurize(train, rc), ytr)
            rand_scores.append(fidelity(c.predict(featurize(test, rc)), yte))
    fid_rand = float(np.mean(rand_scores)) if rand_scores else float("nan")

    used = sorted({int(f) for f in clf.tree_.feature if f >= 0})
    raw1, rule1, cl1 = tree_to_rule(clf, names, target=1)
    raw0, rule0, cl0 = tree_to_rule(clf, names, target=0)

    L = ["=" * 72, "BOOLEAN RULE OVER MINED SUBGRAPHS", "=" * 72,
         f"concepts mined on : {args.train_mined}",
         f"train graphs      : {len(train)}   test graphs: {len(test)}",
         f"concepts offered  : {len(concepts)}   used by the rule: {len(used)}"
         f"   tree depth: {args.depth}",
         f"target            : the GNN's predicted label", "",
         "-" * 72, "CONCEPTS", "-" * 72]
    for j, (name, pat) in enumerate(concepts):
        mark = " *" if j in used else "  "
        L.append(f" {mark} C{j:<3} [{len(pat)}e] {name}")

    L += ["", "-" * 72, "RULE  (simplified)", "-" * 72,
          f"predict HARMFUL (class 1) if:\n     {rule1}", "",
          f"predict BENIGN (class 0) if:\n     {rule0}", ""]

    if cl1:
        L.append("tree leaves for class 1 before simplification "
                 "(n = train graphs reaching that leaf):")
        for conds, n, pur in cl1:
            L.append(f"   n={n:<5} purity={pur:.2f}   "
                     f"{' AND '.join(conds)}")
        L += ["", "raw (unsimplified) rule for class 1:",
              f"     {raw1}", ""]

    L += ["-" * 72, "FIDELITY  (agreement with the GNN)", "-" * 72,
          f"  train            : {fid_tr:.3f}",
          f"  TEST (held out)  : {fid_te:.3f}",
          f"  majority class   : {fid_maj:.3f}",
          f"  random concepts  : {fid_rand:.3f}"
          + ("" if rand_scores else "  (too few concepts to sample)"), ""]

    gap = fid_te - fid_rand if rand_scores else float("nan")
    if gap == gap and gap > 0.05:
        L.append("=> the mined concepts carry information that randomly "
                 "chosen ones do not.")
    elif gap == gap:
        L.append("=> no better than randomly chosen mined concepts: the "
                 "selection is not adding value.")
    L.append("=" * 72)

    report = "\n".join(L)
    print(report)
    out = args.out or os.path.join(os.path.dirname(args.train_mined),
                                   "rules_report.txt")
    with open(out, "w") as f:
        f.write(report)
    print(f"\nSaved: {out}")
    print("\nRaw tree:\n" + export_text(clf, feature_names=names))


if __name__ == "__main__":
    main()
