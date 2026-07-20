"""Reproducible benchmark: size/accuracy ROI for treegptq vs baselines.

Baselines beyond the package's own methods:
  fp32        the model as shipped (pickle + lzma), the null everyone uses
  fp32_tgq    same values in the .tgq container (isolates container effects)
  fp16        leaf values cast to float16
  prune@match trees truncated (boosting) or subsampled (RF) until the .tgq
              size matches gptq@3, answering "why not just fewer trees?"

Run: python benchmarks/run.py [--suite smoke|full]
Deterministic: fixed seeds, versions printed, results to CSV.
"""

from __future__ import annotations

import argparse
import io
import lzma
import pickle
import sys
import time

import numpy as np
import pandas as pd

import treegptq as tq
from treegptq import adapters
from treegptq.api import _metrics, _tgq_bytes, _write_tgq
from treegptq.ir import EnsembleIR
from treegptq.quant import quantize_leaves

SEED = 0
BITS = [2, 3, 4]
METHODS = ["rtn", "wkmeans", "gptq", "gptq_cb", "gptq_alloc"]


def native_bytes(model) -> int:
    return len(lzma.compress(pickle.dumps(model), preset=6))


def subset_ir(ir: EnsembleIR, k: int, renorm: bool) -> EnsembleIR:
    out = ir.copy()
    out.trees = out.trees[:k]
    w = out.tree_weights[:k].copy()
    if renorm:
        w = w * (len(ir.trees) / k)
    out.tree_weights = w
    return out


def prune_to_bytes(ir: EnsembleIR, target: int, renorm: bool) -> tuple[EnsembleIR, int]:
    lo, hi = 1, len(ir.trees)
    best = ir
    while lo < hi:
        mid = (lo + hi) // 2
        cand = subset_ir(ir, mid, renorm)
        if _tgq_bytes(cand, {}) <= target:
            best, lo = cand, mid + 1
        else:
            hi = mid
    return best, len(best.trees)


def fp16_ir(ir: EnsembleIR) -> EnsembleIR:
    out = ir.copy()
    out.set_leaf_values([v.astype(np.float16).astype(np.float64)
                         for v in out.get_leaf_values()])
    return out


def fp16_bytes(ir: EnsembleIR) -> int:
    buf = io.BytesIO()
    half = ir.copy()
    _write_tgq(buf, half, {})
    # approximate: value arrays re-stored at 16 bits instead of 32
    # exact accounting: rewrite with float16 values
    import numpy as _np

    arrays = {}
    raw = io.BytesIO()
    for t, tree in enumerate(ir.trees):
        arrays[f"f{t}"] = tree.feature.astype(_np.int16)
        arrays[f"t{t}"] = tree.threshold.astype(_np.float32)
        arrays[f"l{t}"] = tree.left.astype(_np.int16)
        arrays[f"r{t}"] = tree.right.astype(_np.int16)
        arrays[f"d{t}"] = _np.packbits(tree.default_left)
        arrays[f"v{t}"] = tree.value[tree.leaf_ids(), 0].astype(_np.float16)
    _np.savez(raw, **arrays)
    return 4 + len(lzma.compress(raw.getvalue(), preset=6))


def load_suite(suite: str):
    from sklearn.datasets import fetch_california_housing, fetch_covtype, make_friedman1
    from sklearn.model_selection import train_test_split

    rng = np.random.RandomState(SEED)
    tasks = []

    Xc, yc = fetch_california_housing(return_X_y=True)
    tasks.append(("california", Xc, yc, "reg"))

    Xf, yf = make_friedman1(n_samples=50000, n_features=10, noise=1.0, random_state=3)
    tasks.append(("friedman1", Xf, yf, "reg"))

    if suite == "full":
        cov = fetch_covtype()
        idx = rng.choice(len(cov.target), 150000, replace=False)
        tasks.append(("covtype_bin", cov.data[idx],
                      (cov.target[idx] == 2).astype(float), "binclf"))

    split = {}
    for name, X, y, kind in tasks:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_tr, y_tr, test_size=min(10000, len(X_tr) // 4), random_state=2)
        split[name] = (X_fit, y_fit, X_cal, y_cal, X_te, y_te, kind)
    return split


def train_models(name, X_fit, y_fit, kind, suite):
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor

    n_trees = 500 if suite == "full" else 300
    models = {}
    obj = "binary:logistic" if kind == "binclf" else "reg:squarederror"
    d = xgb.DMatrix(X_fit, label=y_fit)
    models["xgboost"] = xgb.train(
        {"max_depth": 6, "eta": 0.1, "subsample": 0.8, "colsample_bytree": 0.8,
         "objective": obj, "seed": SEED, "nthread": 8}, d, n_trees)
    if kind == "reg":
        models["rf"] = RandomForestRegressor(
            n_estimators=min(n_trees, 300), max_depth=10, n_jobs=8,
            random_state=SEED).fit(X_fit, y_fit)
    return models


def run(suite: str, only: str = "") -> pd.DataFrame:
    rows = []
    split = load_suite(suite)
    if only:
        split = {k: v for k, v in split.items() if k in only.split(",")}
    for name, (X_fit, y_fit, X_cal, y_cal, X_te, y_te, kind) in split.items():
        for mkind, model in train_models(name, X_fit, y_fit, kind, suite).items():
            t0 = time.time()
            ir = adapters.to_ir(model, np.asarray(X_cal)[:256])
            fp32_native = native_bytes(model)
            fp32_tgq = _tgq_bytes(ir, {})
            m0 = _metrics(ir, X_te, y_te)
            primary = "rmse" if "rmse" in m0 else "logloss"

            def add(method, bits, byts, metrics, extra=""):
                rows.append(dict(dataset=name, model=mkind, method=method, bits=bits,
                                 bytes=byts, native_bytes=fp32_native,
                                 compression_vs_native=fp32_native / byts,
                                 compression_vs_tgq32=fp32_tgq / byts,
                                 **{f"m_{k}": v for k, v in metrics.items()},
                                 rel_delta_pct=100 * (metrics[primary] - m0[primary])
                                 / abs(m0[primary]), extra=extra))

            add("fp32_native", 32, fp32_native, m0)
            add("fp32_tgq", 32, fp32_tgq, m0)
            h = fp16_ir(ir)
            add("fp16", 16, fp16_bytes(ir), _metrics(h, X_te, y_te))

            gptq3_bytes = None
            for b in BITS:
                for meth in METHODS:
                    qir, info = quantize_leaves(ir, X_cal, bits=b, method=meth, seed=SEED)
                    byts = _tgq_bytes(qir, info)
                    if meth == "gptq" and b == 3:
                        gptq3_bytes = byts
                    add(meth, info["avg_bits"], byts, _metrics(qir, X_te, y_te))

            pruned, k = prune_to_bytes(ir, gptq3_bytes, renorm=(mkind == "rf"))
            add("prune@gptq3", 32, _tgq_bytes(pruned, {}),
                _metrics(pruned, X_te, y_te), extra=f"kept {k}/{len(ir.trees)} trees")
            print(f"{name}/{mkind} done in {time.time()-t0:.0f}s", file=sys.stderr)
    return pd.DataFrame(rows)


def roi_table(df: pd.DataFrame) -> str:
    lines = []
    for (ds, mk), g in df.groupby(["dataset", "model"]):
        base = g[g.method == "fp32_native"].iloc[0]
        primary = "m_rmse" if "m_rmse" in g and g["m_rmse"].notna().any() else "m_logloss"
        lines.append(f"\n{ds} / {mk}  (fp32: {base['bytes']:,} bytes, "
                     f"{primary[2:]}={base[primary]:.4f})")
        for _, r in g.sort_values(["bits", "method"]).iterrows():
            if r.method.startswith("fp32"):
                continue
            lines.append(f"  {r.method:12s} {r.bits:>5.3g}b  "
                         f"{r.compression_vs_native:5.1f}x smaller  "
                         f"{primary[2:]} {r[primary]:.4f} ({r.rel_delta_pct:+.2f}%)  {r.extra}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--datasets", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    import sklearn
    import xgboost

    print(f"treegptq {tq.__version__} | numpy {np.__version__} | "
          f"xgboost {xgboost.__version__} | sklearn {sklearn.__version__} | seed {SEED}")
    df = run(args.suite, args.datasets)
    out = args.out or f"benchmarks/results_{args.suite}.csv"
    df.to_csv(out, index=False)
    print(roi_table(df))
    print(f"\nresults -> {out}")
