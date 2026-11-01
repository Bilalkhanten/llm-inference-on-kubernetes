#!/usr/bin/env python3
"""KV-cache capacity calculator. All inputs are assumptions you can override."""
import argparse

GiB = 1024**3
# name: (params_B, layers, kv_heads, head_dim)
MODELS = {
    "Mistral-7B-v0.3":   (7.25, 32, 8, 128),
    "Llama-3.1-8B":      (8.03, 32, 8, 128),
    "Qwen2.5-32B":       (32.5, 64, 8, 128),
    "Llama-3.1-70B":     (70.6, 80, 8, 128),
    "Llama-3.1-405B":    (405.0, 126, 8, 128),
}

def kv_bytes_per_token(layers, kv_heads, head_dim, dtype_bytes):
    return 2 * layers * kv_heads * head_dim * dtype_bytes   # K and V

def plan(name, tp, gpu_gib, util, w_bytes, kv_bytes, overhead_gib, ctx):
    params, L, H, D = MODELS[name]
    weights_gib = params * 1e9 * w_bytes / GiB
    per_gpu_weights = weights_gib / tp
    budget = gpu_gib * util - per_gpu_weights - overhead_gib      # per GPU, GiB
    if budget <= 0:
        return None
    # KV heads are sharded across TP ranks, so per-token KV cost per GPU is /tp
    per_tok = kv_bytes_per_token(L, H, D, kv_bytes) / tp
    tokens = budget * GiB / per_tok
    return dict(model=name, tp=tp, weights_gib=weights_gib, kv_per_tok_kib=kv_bytes_per_token(L,H,D,kv_bytes)/1024,
                budget_gib=budget, tokens=tokens, conc=tokens/ctx)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-gib", type=float, default=80)
    ap.add_argument("--util", type=float, default=0.90)
    ap.add_argument("--overhead-gib", type=float, default=3.0, help="activations + cuda graphs + NCCL buffers, per GPU (ASSUMPTION)")
    ap.add_argument("--ctx", type=int, default=8192, help="average live context per sequence")
    a = ap.parse_args()
    print(f"GPU={a.gpu_gib:.0f}GiB  util={a.util}  overhead={a.overhead_gib}GiB/GPU  avg_ctx={a.ctx}  bf16 weights")
    print(f"{'model':<16}{'TP':>3} {'KV/tok':>8} {'weights/GPU':>12} {'KV budget':>10} {'KV tokens':>10} {'seqs bf16':>10} {'seqs fp8':>9}")
    for name, tp in [("Mistral-7B-v0.3",1),("Llama-3.1-8B",1),("Qwen2.5-32B",1),("Qwen2.5-32B",2),("Llama-3.1-70B",1),("Llama-3.1-70B",2),("Llama-3.1-70B",4),("Llama-3.1-405B",8)]:
        b = plan(name, tp, a.gpu_gib, a.util, 2, 2, a.overhead_gib, a.ctx)
        f = plan(name, tp, a.gpu_gib, a.util, 2, 1, a.overhead_gib, a.ctx)
        if not b:
            w = MODELS[name][0]*1e9*2/GiB/tp
            print(f"{name:<16}{tp:>3}  does not fit: {w:.0f} GiB weights/GPU > {a.gpu_gib*a.util:.0f} GiB budget")
            continue
        print(f"{name:<16}{tp:>3} {b['kv_per_tok_kib']:>5.0f}KiB {b['weights_gib']/tp:>8.1f} GiB {b['budget_gib']:>7.1f} GiB {b['tokens']:>10,.0f} {b['conc']:>10.1f} {f['conc']:>9.1f}")
