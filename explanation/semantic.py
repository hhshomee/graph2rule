def safe_float(x, default=0.0):
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        if v in (float("inf"), float("-inf")):
            return default
        return v
    except Exception:
        return default


def node_identity(raw_node):
    ntype = raw_node.get("node_type", "unknown")
    layer = int(round(safe_float(raw_node.get("layer_depth", 0))))
    if ntype == "head":
        idx = int(round(safe_float(raw_node.get("head_idx", 0))))
    elif ntype == "ffn":
        raw_idx = raw_node.get("neuron_idx", raw_node.get("ffn_idx", 0))
        idx = int(round(safe_float(raw_idx, 0)))
    else:
        idx = -1
    return (ntype, layer, idx)
def identity_label(identity):
    ntype, layer, idx = identity
    tag = "H" if ntype == "head" else ("F" if ntype == "ffn" else "?")
    if ntype == "ffn" and idx == -1:
        return f"F-L{layer}"
    return f"{tag}-L{layer}-{idx}"