"""
AE Reconstruction Probe
========================
Thử nghiệm: lấy hidden states thực tế từ LM và dùng QAAE để reconstruct,
sau đó so sánh token predictions giữa original và reconstructed.

Pipeline:
  prompt  →  LM forward (layer 0..L)  →  h_orig [seq, D]  (layer -1 output)
                                                 ↓
                                         QAAE(h_latent=h_orig[t],
                                              h_input=h_orig[t-k])
                                                 ↓ h_rec [seq, D]
  Compare:
    lm_head(h_orig)  vs  lm_head(h_rec)  → top-k tokens per position
    greedy decode from last position using original vs reconstructed state

Usage:
    /venv/latentmas/bin/python3 probe_ae_infer.py \
        --model Qwen/Qwen3-4B \
        --layer -1 \
        --prompt "The patient presents with a fever and" \
        --gen_tokens 40
"""

import argparse
import os
import sys
import math

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from methods.query_aware_ae import QueryAwareAutoencoder


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_ae(checkpoint_dir: str, layer_idx: int, device: str) -> QueryAwareAutoencoder:
    ckpt_path = os.path.join(checkpoint_dir, f"ae_layer_{layer_idx}.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt["config"]
    model = QueryAwareAutoencoder(
        hidden_dim     = cfg["hidden_dim"],
        bottleneck_dim = cfg["bottleneck_dim"],
        gate_hidden    = cfg.get("gate_hidden"),
        encoder_layers = cfg.get("encoder_layers", 2),
        decoder_layers = cfg.get("decoder_layers", 3),
        mlp_hidden     = cfg.get("mlp_hidden", 2048),
    ).to(device)
    sd_key = "state_dict" if "state_dict" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[sd_key])
    model.eval()
    print(f"  [AE] Loaded layer {layer_idx} checkpoint: {ckpt_path}")
    return model


def _hook_hidden_states(hf_model, target_layer_abs: int):
    """Attach a forward hook to extract hidden states at target_layer_abs."""
    captured = {}

    def hook_fn(module, input, output):
        # output is (hidden_states, ...) for decoder layers
        hs = output[0] if isinstance(output, tuple) else output
        captured["hidden_states"] = hs.detach().float()

    layers = hf_model.model.layers
    handle = layers[target_layer_abs].register_forward_hook(hook_fn)
    return captured, handle


def _top_k_tokens(logits_1d: torch.Tensor, tokenizer, k: int = 10):
    probs = F.softmax(logits_1d.float(), dim=-1)
    top = probs.topk(k)
    results = []
    for prob, idx in zip(top.values.tolist(), top.indices.tolist()):
        tok = tokenizer.decode([idx], skip_special_tokens=False)
        results.append((tok, prob, idx))
    return results


def _greedy_decode(
    hf_model,
    tokenizer,
    input_ids: torch.Tensor,
    initial_hidden: torch.Tensor,   # [1, D] — to replace last-position hidden state
    target_layer_abs: int,
    n_tokens: int,
    device: str,
) -> str:
    """Generate n_tokens greedily.
    At the FIRST step, we inject initial_hidden as the hidden state of the
    last token (by hooking the output of target_layer_abs and replacing
    the last position). Subsequent tokens are generated normally.
    """
    generated_ids = input_ids.clone()
    injected = [False]

    def injection_hook(module, inp, output):
        if injected[0]:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        # Replace last position with our custom hidden state
        hs = hs.clone()
        hs[:, -1, :] = initial_hidden.to(hs.device, hs.dtype)
        injected[0] = True
        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    handle = hf_model.model.layers[target_layer_abs].register_forward_hook(injection_hook)

    out_tokens = []
    try:
        with torch.no_grad():
            past_kv = None
            ids = generated_ids
            for step in range(n_tokens):
                out = hf_model(
                    input_ids=ids,
                    past_key_values=past_kv,
                    use_cache=True,
                    output_hidden_states=False,
                )
                logits   = out.logits[:, -1, :]    # [1, vocab]
                past_kv  = out.past_key_values
                next_id  = logits.argmax(dim=-1)   # greedy
                out_tokens.append(next_id.item())
                ids = next_id.unsqueeze(0)
                injected[0] = True   # only inject at step 0
    finally:
        handle.remove()

    return tokenizer.decode(out_tokens, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          default="Qwen/Qwen3-4B")
    parser.add_argument("--layer",          type=int, default=-1,
                        help="Signed layer index of AE (-1 = last layer)")
    parser.add_argument("--checkpoint_dir", default="ae_checkpoints/paired_hidden_state_cache")
    parser.add_argument("--prompt",
                        default="""You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.

Question: 
A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?
[
{
"key": "A",
"value": "Disclose the error to the patient but leave it out of the operative report"
},
{
"key": "B",
"value": "Disclose the error to the patient and put it in the operative report"
},
{
"key": "C",
"value": "Tell the attending that he cannot fail to disclose this mistake"
},
{
"key": "D",
"value": "Report the physician to the ethics committee"
},
{
"key": "E",
"value": "Refuse to dictate the operative report"
}
]

Your outlined plan should be concise with a few bulletpoints for each step. Do not produce the final answer.
Now output your plan to solve the question below:""")
    parser.add_argument("--gen_tokens",     type=int, default=40,
                        help="Number of tokens to greedily generate for text comparison")
    parser.add_argument("--topk",           type=int, default=8,
                        help="Top-k tokens to show in per-position comparison")
    parser.add_argument("--input_offset",   type=int, default=5,
                        help="k: h_input = hidden state of (last_pos - k) token")
    parser.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint_dir = os.path.join(ROOT, args.checkpoint_dir)
    device = args.device

    # ── 1. Load LM ────────────────────────────────────────────────────────
    print(f"\n[1] Loading LM: {args.model} ...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    hf_model.eval()

    num_layers = hf_model.config.num_hidden_layers
    layer_abs  = num_layers + args.layer if args.layer < 0 else args.layer
    print(f"   Total transformer layers: {num_layers}")
    print(f"   Target layer: {args.layer} → abs index {layer_abs}")

    # ── 2. Load QAAE ──────────────────────────────────────────────────────
    print(f"\n[2] Loading QAAE ...")
    ae = _load_ae(checkpoint_dir, args.layer, device)

    # ── 3. Tokenize prompt ────────────────────────────────────────────────
    print(f"\n[3] Prompt:\n   \"{args.prompt}\"")
    enc      = tokenizer(args.prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]                       # [1, T]
    T = input_ids.shape[1]
    print(f"   Tokens: {T}")

    # ── 4. Forward pass with hook → capture hidden states ─────────────────
    print(f"\n[4] Running LM forward to capture hidden states at layer {layer_abs} ...")
    captured, handle = _hook_hidden_states(hf_model, layer_abs)
    with torch.no_grad():
        out = hf_model(**enc, output_hidden_states=False)
    handle.remove()

    H = captured["hidden_states"].squeeze(0).float()  # [T, D]
    print(f"   Captured H: {list(H.shape)}")

    # ── 5. Build h_latent / h_input pairs ─────────────────────────────────
    # Strategy: for each token position t, treat:
    #   h_latent = H[t]                  (token being analysed)
    #   h_input  = H[max(0, t - offset)] (context token = "query")
    print(f"\n[5] Computing QAAE reconstructions (input_offset={args.input_offset}) ...")

    offset = min(args.input_offset, T - 1)
    h_latent = H.to(device)                                          # [T, D]
    h_input  = torch.cat([H[:offset], H[:T - offset]], dim=0).to(device)  # [T, D]
    # simpler: h_input_t = H[max(0, t-offset)]
    idx_input = torch.clamp(
        torch.arange(T, device=device) - offset, min=0
    )
    h_input = H[idx_input].to(device)                               # [T, D]

    with torch.no_grad():
        # normalize=True uses stored norm stats from training
        h_rec, z, gate = ae(h_latent, h_input, normalize=True)   # [T, D]

    mse = (h_rec - h_latent).pow(2).mean().item()
    cos = F.cosine_similarity(h_rec, h_latent, dim=-1).mean().item()
    print(f"   Reconstruction MSE:      {mse:.5f}")
    print(f"   Cosine similarity (avg): {cos:.4f}")
    print(f"   Gate mean:               {gate.mean().item():.4f}")
    print(f"   Gate > 0.5 fraction:     {(gate > 0.5).float().mean().item():.4f}")

    # ── 6. Per-position top-k token comparison ─────────────────────────────
    print(f"\n[6] Token-level prediction comparison (top-{args.topk}) ...")

    lm_head = hf_model.lm_head
    # Some models have a final norm before lm_head
    final_norm = getattr(hf_model.model, "norm", None)

    def _decode_h(h_select: torch.Tensor) -> torch.Tensor:
        """Apply final norm + lm_head to get logits. [T, D] → [T, vocab]"""
        h = h_select.to(device, dtype=hf_model.dtype)
        if final_norm is not None:
            h = final_norm(h)
        return lm_head(h).float()

    with torch.no_grad():
        logits_orig = _decode_h(h_latent)   # [T, vocab]
        logits_rec  = _decode_h(h_rec)      # [T, vocab]

    # Show last 5 positions
    positions_to_show = list(range(max(0, T - 5), T))
    tokens_in_prompt  = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    print()
    print("─" * 72)
    for pos in positions_to_show:
        tok_str = tokens_in_prompt[pos] if pos < len(tokens_in_prompt) else "?"
        print(f"\n  Position {pos}  |  token = '{tok_str}'")

        orig_top = _top_k_tokens(logits_orig[pos], tokenizer, args.topk)
        rec_top  = _top_k_tokens(logits_rec[pos],  tokenizer, args.topk)

        print(f"  {'ORIGINAL':^34}  {'RECONSTRUCTED':^34}")
        print(f"  {'token':<18} {'prob':>8}    {'token':<18} {'prob':>8}")
        print(f"  {'-'*30}    {'-'*30}")
        for (ot, op, _), (rt, rp, _) in zip(orig_top, rec_top):
            match = "✓" if ot == rt else " "
            print(f"  {ot!r:<18} {op:>8.4f}  {match}  {rt!r:<18} {rp:>8.4f}")
    print("─" * 72)

    # ── 7. Greedy decode comparison ────────────────────────────────────────
    print(f"\n[7] Greedy text generation ({args.gen_tokens} tokens) ...")
    print(f"    Starting from last position hidden state ...")

    h_last_orig = h_latent[-1:].float()   # [1, D]
    h_last_rec  = h_rec[-1:].float()       # [1, D]

    print(f"\n  Prompt: \"{args.prompt}\"")

    text_orig = _greedy_decode(
        hf_model, tokenizer, input_ids,
        h_last_orig, layer_abs, args.gen_tokens, device
    )
    text_rec  = _greedy_decode(
        hf_model, tokenizer, input_ids,
        h_last_rec,  layer_abs, args.gen_tokens, device
    )

    print(f"\n  ┌─ ORIGINAL hidden state ──────────────────────────────────")
    print(f"  │ {text_orig}")
    print(f"  └──────────────────────────────────────────────────────────")
    print(f"\n  ┌─ RECONSTRUCTED (QAAE) ───────────────────────────────────")
    print(f"  │ {text_rec}")
    print(f"  └──────────────────────────────────────────────────────────")

    # token-level agreement
    orig_ids = tokenizer.encode(text_orig, add_special_tokens=False)
    rec_ids  = tokenizer.encode(text_rec,  add_special_tokens=False)
    agree = sum(a == b for a, b in zip(orig_ids, rec_ids))
    total = max(len(orig_ids), len(rec_ids), 1)
    print(f"\n  Token agreement: {agree}/{min(len(orig_ids),len(rec_ids))} "
          f"= {agree/total:.1%}")
    print()


if __name__ == "__main__":
    main()
