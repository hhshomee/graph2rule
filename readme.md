# Graph2Rule

**Graph2Rule: Learning Behavioral Rules from Transformer Computation Graphs**.

The pipeline has four steps:

1. Build one computation graph per prompt from a frozen transformer.
2. Train a graph classifier (GCN / GAT) and an MLP baseline on these graphs.
3. Explain the trained classifier with GNNExplainer and mine recurring class-discriminative motifs.
4. Turn the motifs into a Boolean rule and measure its test fidelity to the classifier.

<!-- ![Graph2Rule pipeline](pipeline.png) -->

*Overview of Graph2Rule ([PDF version](pipeline.pdf)).*

---

## Repository structure

```
Graph2Rule/
├── pipeline.pdf     # pipeline figure 
├── data/            # fixed train/test splits (CSV: index, prompt, label)
├── graph/           # step 1: graph construction
│   ├── tensors.py
│   └── save_graph.py
├── classification/  # step 2: graph classifiers
│   ├── gnn.py
│   └── mlp.py
├── explanation/     # steps 3-4: explanation, mining, rules
│   ├── explainer.py
│   ├── semantic.py
│   ├── subgraph_mining.py
│   └── rules.py
└── results/         # example training logs
```

---

## Setup

```bash
pip install torch torch_geometric transformers scikit-learn pandas numpy networkx gspan-mining
```

The scripts import each other across folders, so run every command from the repository root after setting:

```bash
export PYTHONPATH=$PWD/graph:$PWD/classification:$PWD/explanation
```

Llama and Gemma models are gated on Hugging Face; log in first with `huggingface-cli login`.

---

## Step 1: Build graphs

Run once per dataset, split, and backbone.

```bash
python graph/tensors.py \
    --in_csv data/got_train_all_unique.csv \
    --out_dir graphs/got/got_train_all_unique_llama3.2_1b \
    --model_name meta-llama/Llama-3.2-1B-Instruct \
    --edge_weight_mode pretrained
```

Repeat with `data/got_test_all_unique.csv` and a matching `..._test_...` output folder.

| Backbone | `--model_name` |
|---|---|
| T5 | `t5-small` |
| Llama-1B | `meta-llama/Llama-3.2-1B-Instruct` |
| Llama-3B | `meta-llama/Llama-3.2-3B-Instruct` |
| Gemma-1B | `google/gemma-3-1b-it` |

Datasets: `got`, `hc3`, `wildjailbreak`. Each graph is saved as `graph_<index>.json`.

---

## Step 2: Train classifiers

**GCN / GAT** (`--model gcn` or `--model gat`; we use seeds 1-4):

```bash
python classification/gnn.py \
    --train_dir graphs/got/got_train_all_unique_llama3.2_1b \
    --test_dir  graphs/got/got_test_all_unique_llama3.2_1b \
    --run_name got_gat_seed1_llama3.2_1b \
    --model gat \
    --seed 1 \
    --epochs 50 --val_ratio 0.15 --batch_size 16 --hidden 128 \
    --lr 1e-3 --weight_decay 1e-4 --dropout 0.1
```

Saves the checkpoint to `results/<run_name>.pt` and test predictions to `results/<run_name>.csv`.

**MLP baseline:**

```bash
python classification/mlp.py \
    --train_dir graphs/got/got_train_all_unique_llama3.2_1b \
    --test_dir  graphs/got/got_test_all_unique_llama3.2_1b \
    --epochs 50 --val_ratio 0.15 --batch_size 16 --lr 1e-3 --dropout 0.1
```

Test accuracy, precision, recall, and macro-F1 are printed at the end of each run.

---

## Step 3: Explain and mine motifs

**3a. Explanation subgraphs.** Run GNNExplainer (2 initializations, 100 epochs) and keep the top 120 edges per graph. Do this for both the train and test graphs:

```bash
RUN=got_gat_seed1_llama3.2_1b

python explanation/subgraph_mining.py dump \
    --ckpt results/$RUN.pt \
    --graph_dir graphs/got/got_train_all_unique_llama3.2_1b \
    --out mining/$RUN/train/explanation_subgraphs_full.json \
    --seeds 2 --epochs 100 --top_k_edges 120

python explanation/subgraph_mining.py dump \
    --ckpt results/$RUN.pt \
    --graph_dir graphs/got/got_test_all_unique_llama3.2_1b \
    --out mining/$RUN/test/explanation_subgraphs_full.json \
    --seeds 2 --epochs 100 --top_k_edges 120
```

**3b. Mine motifs** with gSpan, on the training explanations only. FFN neurons are grouped into blocks of 1024, the minimum support is 10%, and motifs have at most 6 nodes:

```bash
python explanation/subgraph_mining.py mine \
    --in mining/$RUN/train/explanation_subgraphs_full.json \
    --granularity block --block_size 1024 --engine gspan \
    --aggregate correct --min_support 0.10 \
    --max_nodes 6 --max_edges 6 --closure minimal \
    --n_perm 1000 --top_k 20 \
    --out_dir mining/$RUN/train
```

Output: `mining/$RUN/train/mined_subgraphs_block1024_gspan.csv`, the ranked motifs with their class-wise frequencies and contrast Δ(P).

---

## Step 4: Rules and test fidelity

Take the top 15 motifs as concepts, fit a depth-3 decision tree to the GNN's predictions on the training set, and evaluate fidelity on the test set:

```bash
python explanation/rules.py \
    --train_mined mining/$RUN/train/mined_subgraphs_block1024_gspan.csv \
    --train_dump  mining/$RUN/train/explanation_subgraphs_full.json \
    --test_dump   mining/$RUN/test/explanation_subgraphs_full.json \
    --block_size 1024 --top_k 15 --depth 3
```

This prints the concepts, the simplified Boolean rule, and the train/test fidelity against majority-class and random-concept baselines. The report is saved as `rules_report.txt` next to the mined CSV.

---

## Data format

Each CSV in `data/` has three columns: `index` (sample id, also used as the graph filename), `prompt`, and `label` (0/1). All experiments use the same fixed train/test split.
