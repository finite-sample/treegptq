"""Leaf-value quantization passes over the IR.

Methods (bits b per leaf value):
  rtn            per-tree min-max uniform grid, round to nearest
  rtn_global     one global uniform grid
  kmeans         global codebook (2^b entries), unweighted 1D k-means
  wkmeans        global codebook, weights = calib traffic x hessian mass
  gptq           sequential error compensation onto the per-tree grid,
                 with the stability fix: compensation clipped to the tree's
                 leaf range, targets quantized on the ORIGINAL grid
  gptq_cb        sequential compensation onto the weighted codebook
  gptq_alloc     gptq + per-tree bit allocation (avg b bits, floor 2, cap 8)

All methods are post-training and need only a calibration sample. Compensation
processes trees in ensemble order; before quantizing tree t, each leaf value is
shifted by the hessian-weighted mean accumulated quantization error of the
calibration rows in that leaf, so tree t cancels its predecessors' error.
"""

from __future__ import annotations

import numpy as np

from .ir import EnsembleIR

METHODS = ("rtn", "rtn_global", "kmeans", "wkmeans", "gptq", "gptq_cb", "gptq_alloc")


# ---------------------------------------------------------------- helpers

def _rtn(v: np.ndarray, lo: float, hi: float, b: int) -> np.ndarray:
    if hi <= lo:
        return v.copy()
    levels = 2**b - 1
    q = np.round((v - lo) / (hi - lo) * levels)
    return lo + np.clip(q, 0, levels) / levels * (hi - lo)


def _snap(v: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    return codebook[np.argmin(np.abs(v[:, None] - codebook[None, :]), axis=1)]


def _kmeans_1d(values: np.ndarray, weights: np.ndarray, k: int,
               n_iter: int = 50, seed: int = 0) -> np.ndarray:
    """Weighted 1D k-means via Lloyd on sorted quantile init. numpy-only."""
    uniq = np.unique(values)
    if len(uniq) <= k:
        return uniq
    rng = np.random.RandomState(seed)
    qs = np.linspace(0, 1, k + 2)[1:-1]
    centers = np.quantile(values, qs) + rng.normal(0, 1e-9, k)
    w = np.maximum(weights, 1e-12)
    for _ in range(n_iter):
        assign = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        new = centers.copy()
        for j in range(k):
            m = assign == j
            if m.any():
                new[j] = np.average(values[m], weights=w[m])
        if np.allclose(new, centers, atol=1e-12):
            centers = new
            break
        centers = new
    return np.sort(np.unique(centers))


def _hessians(margin: np.ndarray, objective: str) -> np.ndarray:
    if objective == "binary_logistic":
        p = 1.0 / (1.0 + np.exp(-margin))
        return np.maximum(p * (1.0 - p), 1e-6)
    return np.ones_like(margin)


def _leaf_traffic_hessian_weights(ir: EnsembleIR, assign: np.ndarray) -> list[np.ndarray]:
    """Per tree: per-leaf weight = sum of hessians of calib rows in the leaf."""
    margin = ir.margin(None, assign=assign) if False else None  # noqa: F841
    m = np.full(assign.shape[0], ir.base_margin)
    for t, tree in enumerate(ir.trees):
        m = m + ir.tree_weights[t] * tree.value[assign[:, t], 0]
    h = _hessians(m, ir.objective)
    out = []
    for t, tree in enumerate(ir.trees):
        lid = tree.leaf_ids()
        pos = np.searchsorted(lid, assign[:, t])
        out.append(np.bincount(pos, weights=h, minlength=len(lid)))
    return out


# ---------------------------------------------------------------- main pass

def quantize_leaves(ir: EnsembleIR, X_cal: np.ndarray, bits: int = 3,
                    method: str = "gptq", seed: int = 0) -> tuple[EnsembleIR, dict]:
    """Quantize leaf values of `ir` (n_out == 1). Returns (new_ir, info).

    info: {"bits_per_tree": array, "avg_bits": float, "codebook": array | None,
           "codes": list of per-tree integer code arrays or None}
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    if ir.n_out != 1:
        raise NotImplementedError("v0.1 quantizes scalar-leaf ensembles")

    out = ir.copy()
    T = len(out.trees)
    assign = out.leaf_assignment(np.asarray(X_cal, dtype=np.float64))
    vals = [v[:, 0] for v in out.get_leaf_values()]
    allv = np.concatenate(vals)
    info: dict = {"codebook": None, "codes": None}
    bits_t = np.full(T, bits)

    if method == "rtn_global":
        lo, hi = float(allv.min()), float(allv.max())
        qvals = [_rtn(v, lo, hi, bits) for v in vals]

    elif method == "rtn":
        qvals = [_rtn(v, float(v.min()), float(v.max()), bits) for v in vals]

    elif method in ("kmeans", "wkmeans", "gptq_cb"):
        if method == "kmeans":
            w = np.ones_like(allv)
        else:
            w = np.concatenate(_leaf_traffic_hessian_weights(out, assign))
        cb = _kmeans_1d(allv, w, 2**bits, seed=seed)
        info["codebook"] = cb
        if method in ("kmeans", "wkmeans"):
            qvals = [_snap(v, cb) for v in vals]
        else:
            qvals = _compensated(out, assign, vals, codebook=cb, bits_t=None)

    elif method == "gptq":
        qvals = _compensated(out, assign, vals, codebook=None, bits_t=bits_t)

    elif method == "gptq_alloc":
        rms = np.array([float(np.sqrt(np.mean(v**2))) + 1e-12 for v in vals])
        bits_t = np.clip(np.round(bits + np.log2(rms) - np.mean(np.log2(rms))), 2, 8).astype(int)
        qvals = _compensated(out, assign, vals, codebook=None, bits_t=bits_t)

    info["bits_per_tree"] = bits_t
    info["avg_bits"] = float(np.mean(bits_t))
    if info["codebook"] is not None:
        info["codes"] = [np.searchsorted(info["codebook"], q) for q in qvals]
    out.set_leaf_values([q[:, None] for q in qvals])
    return out, info


def _compensated(ir: EnsembleIR, assign: np.ndarray, vals: list[np.ndarray],
                 codebook: np.ndarray | None, bits_t: np.ndarray | None) -> list[np.ndarray]:
    """Sequential error compensation with the bounded-grid stability fix."""
    n = assign.shape[0]
    acc_err = np.zeros(n)                      # accumulated margin error
    margin = np.full(n, ir.base_margin)        # running quantized margin
    qvals = []
    for t, tree in enumerate(ir.trees):
        w_t = float(ir.tree_weights[t])
        lid = tree.leaf_ids()
        pos = np.searchsorted(lid, assign[:, t])
        h = _hessians(margin, ir.objective)
        num = np.bincount(pos, weights=h * acc_err, minlength=len(lid))
        den = np.bincount(pos, weights=h, minlength=len(lid))
        # delta in VALUE space: margin delta / tree weight
        delta = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0) / max(w_t, 1e-12)
        r = max(float(vals[t].max() - vals[t].min()), 1e-12)
        target = vals[t] - np.clip(delta, -r, r)
        if codebook is not None:
            q = _snap(target, codebook)
        else:
            lo, hi = float(vals[t].min()), float(vals[t].max())
            q = _rtn(np.clip(target, lo, hi), lo, hi, int(bits_t[t]))
        qvals.append(q)
        acc_err += w_t * (q[pos] - vals[t][pos])
        margin += w_t * q[pos]
    return qvals
