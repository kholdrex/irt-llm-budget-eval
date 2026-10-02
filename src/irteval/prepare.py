"""Download the public response data and convert it into compact model x item 0/1 matrices.

D1  Open LLM Leaderboard v1 correctness released with tinyBenchmarks (repository licence: MIT),
    file tutorials/data/lb.pickle at a pinned commit.
D2  EmbedLLM correctness data (Apache-2.0), train/val/test CSV files at a pinned revision.

Output: <workdir>/data/{openllm,embedllm}.npz with
    Y       int8 [n_models, n_items]: 1 correct, 0 incorrect, -1 missing
    models  model identifiers
    bench   benchmark of each item
    sub     finer label (MMLU subject for D1, task variants for D2)
The 2.7 GB EmbedLLM training file is streamed and never written to disk.
"""
import csv
import hashlib
import io
import pickle
import sys
import urllib.request

import numpy as np

from .paths import DATA, ensure_dirs

TINYBENCH_URL = ("https://raw.githubusercontent.com/felipemaiapolo/tinyBenchmarks/"
                 "ce9f7024248685ff1a5b507b955051e9a0e6e074/tutorials/data/lb.pickle")
TINYBENCH_SHA256 = "34f44d6a819512ef74d00a95288d252fa679288a10ca167cd97fdbc3aae66437"
EMBEDLLM_URL = "https://huggingface.co/datasets/RZ412/EmbedLLM/resolve/266ae3e180ccb533c86afc9cfb2f5db99d171aea/{}"
EMBEDLLM_ROWS = {"val.csv": 383_488, "test.csv": 388_528, "train.csv": 3_814_272}
TQA_THRESHOLD = 0.5   # TruthfulQA mc2 is continuous in [0, 1]; success = mc2 > 0.5


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _download(url, path, sha256=None):
    """Download url to path once; a cached file is reused only if it passes the checksum."""
    if path.exists() and (sha256 is None or _sha256(path) == sha256):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        if sha256 is not None and _sha256(tmp) != sha256:
            raise RuntimeError(f"checksum mismatch for {path.name}")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def prepare_openllm():
    path = _download(TINYBENCH_URL, DATA / "raw" / "lb.pickle", TINYBENCH_SHA256)
    with open(path, "rb") as f:          # checksum verified above
        raw = pickle.load(f)
    models = [m.replace("open-llm-leaderboard/details_", "") for m in raw["models"]]
    blocks, bench, sub = [], [], []
    for key, val in raw["data"].items():
        c = val["correctness"]               # items x models
        if key.startswith("harness_hendrycksTest_"):
            b, s = "mmlu", key[len("harness_hendrycksTest_"):-2]
        else:
            b = s = key.split("_")[1]
        if b == "truthfulqa":
            c = (c > TQA_THRESHOLD).astype(float)
        blocks.append(c.T)
        bench += [b] * c.shape[0]
        sub += [s] * c.shape[0]
    Y = np.concatenate(blocks, axis=1).astype(np.int8)
    np.savez_compressed(DATA / "openllm.npz", Y=Y, models=np.array(models),
                        bench=np.array(bench), sub=np.array(sub))
    print(f"openllm: {Y.shape[0]} models x {Y.shape[1]} items, mean correctness {Y.mean():.4f}")


def _rows(stream):
    csv.field_size_limit(sys.maxsize)
    reader = csv.reader(stream)
    header = next(reader)
    ip, im, il, icat = (header.index(k) for k in ("prompt_id", "model_id", "label", "category"))
    for row in reader:
        yield int(row[ip]), int(row[im]), int(row[il]), row[icat]


def _family(category):
    return category.split("_")[0] if category.startswith(("mmlu_", "gpqa_")) else category


def prepare_embedllm():
    triples, cats = [], {}
    for name, expected in EMBEDLLM_ROWS.items():
        with urllib.request.urlopen(EMBEDLLM_URL.format(name)) as resp:
            n = 0
            for p, m, label, cat in _rows(io.TextIOWrapper(resp, encoding="utf-8", newline="")):
                triples.append((p, m, label))
                cats.setdefault(p, set()).add(cat)
                n += 1
        if n != expected:
            raise RuntimeError(f"{name}: expected {expected} rows, read {n}")
        print(f"{name}: {n} rows")
    order = _download(EMBEDLLM_URL.format("model_order.csv"), DATA / "raw" / "embedllm_model_order.csv")
    with open(order, encoding="utf-8") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["model_id"]))
    models = [r["model_name"] for r in rows]

    t = np.array(triples, dtype=np.int64)
    item_ids = np.unique(t[:, 0])
    cols = np.searchsorted(item_ids, t[:, 0])
    n_rows = np.zeros((len(models), len(item_ids)), dtype=np.int16)
    n_pos = np.zeros_like(n_rows)
    np.add.at(n_rows, (t[:, 1], cols), 1)
    np.add.at(n_pos, (t[:, 1], cols), t[:, 2].astype(np.int16))
    # the same prompt can be listed under several task variants: majority label, ties -> missing
    Y = np.full(n_rows.shape, -1, dtype=np.int8)
    seen = n_rows > 0
    Y[seen] = (2 * n_pos[seen] > n_rows[seen]).astype(np.int8)
    tie = seen & (2 * n_pos == n_rows)
    Y[tie] = -1
    conflict = (n_pos > 0) & (n_pos < n_rows)
    print(f"cells with several rows {int((n_rows > 1).sum())}, conflicting {int(conflict.sum())}, "
          f"ties set missing {int(tie.sum())}")

    fams = [sorted({_family(c) for c in cats[q]}) for q in item_ids]
    if any(len(f) != 1 for f in fams):
        raise RuntimeError("a prompt is shared by two benchmark families")
    bench = np.array([f[0] for f in fams])
    sub = np.array(["|".join(sorted(cats[q])) for q in item_ids])
    np.savez_compressed(DATA / "embedllm.npz", Y=Y, models=np.array(models), bench=bench, sub=sub,
                        prompt_id=item_ids, n_rows=n_rows.astype(np.int8))
    print(f"embedllm: {Y.shape[0]} models x {Y.shape[1]} items, observed {(Y >= 0).mean():.4f}, "
          f"mean correctness {Y[Y >= 0].mean():.4f}")


def main(which=("openllm", "embedllm")):
    ensure_dirs()
    if "openllm" in which:
        prepare_openllm()
    if "embedllm" in which:
        prepare_embedllm()
