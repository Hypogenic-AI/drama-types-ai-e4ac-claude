"""Extract & cache Qwen activations for GoEmotions texts (for E3 reader modeling)."""
import os, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from extract import load_model, extract, free

def main(model_name, tag, layers, max_texts=None):
    texts = pd.read_parquet("data/ge_texts.parquet")
    if max_texts:
        texts = texts.iloc[:max_texts]
    model, tok, n_layers = load_model(model_name)
    layers = [int(x) for x in layers]
    print(f"extracting {len(texts)} GE texts at layers {layers}")
    acts = extract(model, tok, texts["text"].tolist(), layers, batch_size=64, pooling="mean")
    free(model)
    os.makedirs("data", exist_ok=True)
    out = {f"L{L}": acts[L].astype(np.float16) for L in layers}
    out["text_id"] = texts["text_id"].to_numpy()
    np.savez_compressed(f"data/ge_acts_{tag}.npz", **out)
    print("saved data/ge_acts_%s.npz" % tag, {k: v.shape for k, v in out.items() if k != 'text_id'})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tag", default="qwen7b")
    ap.add_argument("--layers", nargs="+", default=[16, 21, 28])
    ap.add_argument("--max_texts", type=int, default=None)
    a = ap.parse_args()
    main(a.model, a.tag, a.layers, a.max_texts)
