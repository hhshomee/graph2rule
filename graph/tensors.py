import argparse
import os
import torch
import pandas as pd

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from save_graph import build_graph_from_sample, save_nx_graph
class T5Adapter:
    def get_blocks(self, model):
        return model.encoder.block

    def hidden_size(self, model):
        return model.config.d_model

    def num_heads(self, model):
        return model.config.num_heads

    def get_attention_q(self, block):
        return block.layer[0].SelfAttention.q

    def get_attention_o(self, block):
        return block.layer[0].SelfAttention.o

    def get_ffn_wi(self, block):
        return block.layer[1].DenseReluDense.wi

    def get_ffn_wo(self, block):
        return block.layer[1].DenseReluDense.wo

    def forward_backbone(self, model, enc):
        encoder = model.get_encoder()
        return encoder(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )

    def register_ffn_hooks(self, model, ffn_cache):
        handles = []

        def make_hook(layer_idx):
            def hook(module, inputs, output):
                ffn_cache[layer_idx] = output.detach()
            return hook

        for layer_idx, block in enumerate(self.get_blocks(model)):
            handle = self.get_ffn_wi(block).register_forward_hook(make_hook(layer_idx))
            handles.append(handle)

        return handles

    def get_ffn_activations(self, model, ffn_cache):
        return ffn_cache
class LlamaAdapter:
    def get_backbone(self, model):
        # Works for both AutoModel and AutoModelForCausalLM-style wrappers
        return model.model if hasattr(model, "model") else model

    def get_blocks(self, model):
        return self.get_backbone(model).layers

    def hidden_size(self, model):
        return model.config.hidden_size

    def num_heads(self, model):
        return model.config.num_attention_heads

    def get_attention_q(self, block):
        return block.self_attn.q_proj

    def get_attention_o(self, block):
        return block.self_attn.o_proj

    def get_ffn_wi(self, block):
        # LLaMA MLP has gate_proj, up_proj, down_proj.
        # Use up_proj as the T5-like "wi" equivalent for graph compatibility.
        return block.mlp.up_proj

    def get_ffn_wo(self, block):
        return block.mlp.down_proj

    def forward_backbone(self, model, enc):
        backbone = self.get_backbone(model)

        return backbone(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    def register_ffn_hooks(self, model, ffn_cache):
        handles = []

        def make_gate_hook(layer_idx):
            def hook(module, inputs, output):
                if layer_idx not in ffn_cache:
                    ffn_cache[layer_idx] = {}
                ffn_cache[layer_idx]["gate"] = output.detach()
            return hook

        def make_up_hook(layer_idx):
            def hook(module, inputs, output):
                if layer_idx not in ffn_cache:
                    ffn_cache[layer_idx] = {}
                ffn_cache[layer_idx]["up"] = output.detach()
            return hook

        for layer_idx, block in enumerate(self.get_blocks(model)):
            handles.append(block.mlp.gate_proj.register_forward_hook(make_gate_hook(layer_idx)))
            handles.append(block.mlp.up_proj.register_forward_hook(make_up_hook(layer_idx)))

        return handles

    def get_ffn_activations(self, model, ffn_cache):
        """
        For LLaMA/SwiGLU:
            FFN intermediate = act(gate_proj(x)) * up_proj(x)

        This is better than only using up_proj output.
        """
        acts = {}

        for layer_idx, block in enumerate(self.get_blocks(model)):
            if layer_idx not in ffn_cache:
                continue

            gate = ffn_cache[layer_idx].get("gate", None)
            up = ffn_cache[layer_idx].get("up", None)

            if gate is None or up is None:
                continue

            acts[layer_idx] = block.mlp.act_fn(gate) * up

        return acts
class QwenAdapter(LlamaAdapter):
    pass
class GemmaAdapter(LlamaAdapter):
    pass
def get_model_adapter(model_name):
    name = model_name.lower()

    if "t5" in name:
        return T5Adapter()

    if "llama" in name:
        return LlamaAdapter()

    if "qwen" in name:
        return QwenAdapter()
    if "gemma" in name:
        return GemmaAdapter()

    raise NotImplementedError(f"No adapter defined for model: {model_name}")
def extract_weight_dict(model, adapter):
    weights = {}

    d_model = adapter.hidden_size(model)
    num_heads = adapter.num_heads(model)
    head_dim = d_model // num_heads

    for i, block in enumerate(adapter.get_blocks(model)):
        W_q_full = adapter.get_attention_q(block).weight.detach().cpu()
        W_o_full = adapter.get_attention_o(block).weight.detach().cpu()

        head_weights = []

        for h in range(num_heads):
            s, e = h * head_dim, (h + 1) * head_dim

            head_weights.append({
                "W_q": W_q_full[s:e, :],
                "W_o": W_o_full[:, s:e],
            })

        ffn_wi = adapter.get_ffn_wi(block).weight.detach().cpu()
        ffn_wo = adapter.get_ffn_wo(block).weight.detach().cpu()

        weights[i] = {
            "heads": head_weights,
            "ffn_wi": ffn_wi,
            "ffn_wo": ffn_wo,
        }

    return weights

    
def collate(batch,tokenizer,max_length):
    texts=[x['prompt'] for x in batch]
    sample_ids=[int(x['index'])for x in batch]
    labels = [int(x["label"]) for x in batch]

    enc=tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return sample_ids,labels,enc

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--in_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="graphs", help="Where to save per-sample .pt files")
    parser.add_argument("--model_name", type=str, default="t5-small",
                        help='Examples: "t5-small", "google/flan-t5-base", "meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"')
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_rows", type=int, default=None,help="Use only first N rows of the CSV")

    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--topk_heads", type=int, default=4,
                        help="Number of attention heads per layer to keep in the graph")
    parser.add_argument("--topk_ffn", type=int, default=64,
                        help="Number of FFN neurons per layer to keep in the graph")
    parser.add_argument("--no_cross_layer", action="store_true",
                        help="Disable edges from layer l to l+1")
    parser.add_argument("--no_head_to_ffn", action="store_true",
                        help="Disable within-layer head->ffn edges")
    parser.add_argument("--edge_weight_mode", type=str, default="pretrained",
                        choices=["product", "sum","pretrained"],
                        help='How to compute edge weights: "product" (default), "source", or "one"')
    
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df= pd.read_csv(args.in_csv)
    assert {'index','prompt','label'}.issubset(df.columns)
    if args.max_rows is not None:
        df = df.iloc[:args.max_rows]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    adapter = get_model_adapter(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {}

    if (
    "llama" in args.model_name.lower()
    or "qwen" in args.model_name.lower()
    or "gemma" in args.model_name.lower()):
        load_kwargs["attn_implementation"] = "eager"

    model = AutoModel.from_pretrained(
    args.model_name,
    **load_kwargs,
    ).to(device)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()

    weight_dict = extract_weight_dict(model, adapter)
    ffn_cache = {}
    hook_handles = adapter.register_ffn_hooks(model, ffn_cache)



    rows = df[["index", "prompt", "label"]].to_dict("records")
    loader = DataLoader(
        rows,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate(b, tokenizer, args.max_length),
    )

    with torch.no_grad():
        for sample_ids, labels, enc in loader:
            ffn_cache.clear()

            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = adapter.forward_backbone(model, enc)
            
            def to_cpu(x):
                if x is None:
                    return None
                return x.detach().to("cpu").to(dtype)
            hidden_states = [to_cpu(h) for h in outputs.hidden_states]   # list: [B,T,D]
            attentions = [to_cpu(a) for a in outputs.attentions]         # list: [B,H,T,T]


            ffn_acts = adapter.get_ffn_activations(model, ffn_cache)
            ffn_wi_out = {int(k): to_cpu(v) for k, v in ffn_acts.items()}

            # Save per sample
            B = enc["input_ids"].shape[0]
            for b in range(B):
                sid = int(sample_ids[b])
                payload = {
                    "sample_id": sid,
                    "label": int(labels[b]),
                    "input_ids": enc["input_ids"][b].detach().cpu(),
                    "attention_mask": enc["attention_mask"][b].detach().cpu(),
                    "encoder_hidden_states": [h[b] for h in hidden_states],  # list of [T,D]
                    "encoder_attentions": [a[b] for a in attentions],        # list of [H,T,T]
                    "ffn_wi_out": {layer: tensor[b] for layer, tensor in ffn_wi_out.items()},  # dict layer->[T,d_ff]
                }
                G = build_graph_from_sample(
                    payload,
                    weight_dict=weight_dict,
                    topk_heads=args.topk_heads,
                    topk_ffn=args.topk_ffn,
                    cross_layer_edges=not args.no_cross_layer,
                    within_layer_head_to_ffn=not args.no_head_to_ffn,
                    edge_weight_mode=args.edge_weight_mode,
                )
                G.graph["sample_id"] = sid
                G.graph["label"] = int(labels[b])

                out_prefix = os.path.join(args.out_dir, f"graph_{sid}")
                save_nx_graph(G, out_prefix,save_json=True)
                print(f"[saved] graph_{sid}")
    for h in hook_handles:
        h.remove()
    print(f"Saved per-sample graphs to: {args.out_dir}")                          

if __name__ == "__main__":
    main()
