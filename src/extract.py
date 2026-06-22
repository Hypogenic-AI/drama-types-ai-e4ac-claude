"""
Activation extraction for Qwen2.5 instruct models.
Reads hidden states at selected layers, pooled over text tokens.

Used by E1/E2 (controlled corpus) and E3 (GoEmotions texts).
Feeds RAW text (no chat template) by default so we read the model's representation
of the text content itself; supports a persona chat template for the persona probe.
"""
import os, gc, json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_HOME", os.path.abspath(".hf_cache"))
torch.manual_seed(42)


def pick_device():
    # choose the CUDA device with the most free memory
    best, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best, best_free = i, free
    return best


def load_model(name="Qwen/Qwen2.5-7B-Instruct"):
    dev = pick_device()
    torch.cuda.set_device(dev)
    tok = AutoTokenizer.from_pretrained(name, padding_side="right")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, device_map={"": dev},
        output_hidden_states=True, attn_implementation="eager",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"Loaded {name} on cuda:{dev}; {n_layers} layers, hidden={model.config.hidden_size}")
    return model, tok, n_layers


@torch.no_grad()
def extract(model, tok, texts, layers, batch_size=32, pooling="mean",
            persona=None, max_len=64):
    """Return dict: layer_idx -> np.float32 array (N, hidden).
    pooling: 'mean' (masked mean over tokens) or 'last' (last real token).
    persona: if set, wraps each text in a chat template with this system persona
             (the pooled tokens are still the *assistant-visible* user content tokens).
    """
    dev = next(model.parameters()).device
    out = {l: [] for l in layers}
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i:i + batch_size])
        if persona is not None:
            rendered = []
            for t in batch:
                msgs = [{"role": "system", "content": persona},
                        {"role": "user", "content": f'Read this passage:\n"{t}"'}]
                rendered.append(tok.apply_chat_template(msgs, tokenize=False,
                                                        add_generation_prompt=True))
            enc = tok(rendered, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len + 48)
        else:
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len)
        enc = {k: v.to(dev) for k, v in enc.items()}
        hs = model(**enc).hidden_states  # tuple (n_layers+1) of (B,T,H)
        mask = enc["attention_mask"].unsqueeze(-1).float()  # (B,T,1)
        for l in layers:
            h = hs[l]  # (B,T,H)
            if pooling == "mean":
                pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            else:  # last real token
                lengths = enc["attention_mask"].sum(1) - 1
                pooled = h[torch.arange(h.size(0)), lengths]
            out[l].append(pooled.float().cpu().numpy())
    return {l: np.concatenate(v, axis=0) for l, v in out.items()}


def free(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
