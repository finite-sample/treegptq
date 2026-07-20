"""Public API: quantize / sweep / save / load, plus the ROI report."""

from __future__ import annotations

import io
import lzma
import pickle
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import adapters
from .ir import EnsembleIR
from .quant import METHODS, quantize_leaves

# ---------------------------------------------------------------- metrics

def _rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def _metrics(ir: EnsembleIR, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    p = ir.predict(X)
    if ir.objective == "binary_logistic":
        return {"logloss": _logloss(y, p), "auc": _auc(y, p)}
    return {"rmse": _rmse(y, p)}


# ---------------------------------------------------------------- sizes

def _native_bytes(model: Any) -> int:
    """Serialized size of the native model: pickle + lzma (what people ship)."""
    return len(lzma.compress(pickle.dumps(model), preset=6))


def _tgq_bytes(ir: EnsembleIR, info: dict) -> int:
    buf = io.BytesIO()
    _write_tgq(buf, ir, info)
    return buf.getbuffer().nbytes


# ---------------------------------------------------------------- report

@dataclass
class Report:
    method: str
    bits: float
    bytes_before: int
    bytes_after: int
    metrics_before: dict[str, float] = field(default_factory=dict)
    metrics_after: dict[str, float] = field(default_factory=dict)

    @property
    def compression(self) -> float:
        return self.bytes_before / max(self.bytes_after, 1)

    @property
    def metric_deltas(self) -> dict[str, float]:
        out = {}
        for k, v in self.metrics_before.items():
            out[k] = self.metrics_after.get(k, float("nan")) - v
        return out

    def summary(self) -> str:
        parts = [f"{self.method}@{self.bits:g}b: {self.compression:.1f}x smaller "
                 f"({self.bytes_before:,} -> {self.bytes_after:,} bytes)"]
        for k, v in self.metrics_before.items():
            after = self.metrics_after.get(k, float("nan"))
            rel = 100 * (after - v) / abs(v) if v else float("nan")
            parts.append(f"{k} {v:.4f} -> {after:.4f} ({rel:+.2f}%)")
        return " | ".join(parts)


# ---------------------------------------------------------------- API

def quantize(model: Any, X_cal: np.ndarray, y_cal: np.ndarray | None = None,
             bits: int = 3, method: str = "gptq",
             X_eval: np.ndarray | None = None, y_eval: np.ndarray | None = None,
             seed: int = 0) -> tuple[Any, Report]:
    """Quantize a fitted model's leaf values. Returns (patched native model, Report).

    The returned model is the same class as the input with quantized leaf
    values written back; structure, routing, and inference code are unchanged.
    """
    ir = adapters.to_ir(model, np.asarray(X_cal)[:256])
    qir, info = quantize_leaves(ir, X_cal, bits=bits, method=method, seed=seed)
    patched = adapters.patch_back(model, qir)
    report = Report(method=method, bits=info["avg_bits"],
                    bytes_before=_native_bytes(model),
                    bytes_after=_tgq_bytes(qir, info))
    if X_eval is not None and y_eval is not None:
        report.metrics_before = _metrics(ir, np.asarray(X_eval), np.asarray(y_eval))
        report.metrics_after = _metrics(qir, np.asarray(X_eval), np.asarray(y_eval))
    return patched, report


def sweep(model: Any, X_cal: np.ndarray, y_cal: np.ndarray | None,
          X_eval: np.ndarray, y_eval: np.ndarray,
          bits: list[int] = [2, 3, 4], methods: list[str] = list(METHODS),
          seed: int = 0) -> list[Report]:
    """Frontier: every (method, bits) cell evaluated on held-out data."""
    ir = adapters.to_ir(model, np.asarray(X_cal)[:256])
    before_bytes = _native_bytes(model)
    before_metrics = _metrics(ir, np.asarray(X_eval), np.asarray(y_eval))
    out = []
    for b in bits:
        for m in methods:
            qir, info = quantize_leaves(ir, X_cal, bits=b, method=m, seed=seed)
            out.append(Report(method=m, bits=info["avg_bits"],
                              bytes_before=before_bytes,
                              bytes_after=_tgq_bytes(qir, info),
                              metrics_before=before_metrics,
                              metrics_after=_metrics(qir, np.asarray(X_eval),
                                                     np.asarray(y_eval))))
    return out


# ---------------------------------------------------------------- .tgq format

_MAGIC = b"TGQ1"


def _write_tgq(fh, ir: EnsembleIR, info: dict) -> None:
    arrays: dict[str, np.ndarray] = {
        "base_margin": np.array([ir.base_margin]),
        "tree_weights": np.asarray(ir.tree_weights, dtype=np.float32),
        "objective": np.frombuffer(ir.objective.encode(), dtype=np.uint8),
        "decision": np.frombuffer(ir.decision.encode(), dtype=np.uint8),
    }
    for t, tree in enumerate(ir.trees):
        arrays[f"f{t}"] = tree.feature.astype(np.int16)
        arrays[f"t{t}"] = tree.threshold.astype(np.float32)
        arrays[f"l{t}"] = tree.left.astype(np.int16)
        arrays[f"r{t}"] = tree.right.astype(np.int16)
        arrays[f"d{t}"] = np.packbits(tree.default_left)
        if info.get("codebook") is not None and info.get("codes") is not None:
            arrays[f"c{t}"] = info["codes"][t].astype(np.uint8)
        else:
            arrays[f"v{t}"] = tree.value[tree.leaf_ids(), 0].astype(np.float32)
    if info.get("codebook") is not None:
        arrays["codebook"] = np.asarray(info["codebook"], dtype=np.float32)
    raw = io.BytesIO()
    np.savez(raw, **arrays)
    fh.write(_MAGIC)
    fh.write(lzma.compress(raw.getvalue(), preset=6))


def save(model_or_ir: Any, path: str, X_ref: np.ndarray | None = None,
         info: dict | None = None) -> int:
    """Save a compact .tgq artifact. Returns bytes written."""
    if isinstance(model_or_ir, EnsembleIR):
        ir = model_or_ir
    else:
        if X_ref is None:
            raise ValueError("saving a native model requires X_ref for conversion")
        ir = adapters.to_ir(model_or_ir, np.asarray(X_ref)[:256])
    with open(path, "wb") as fh:
        _write_tgq(fh, ir, info or {})
    import os

    return os.path.getsize(path)


def load(path: str) -> EnsembleIR:
    """Load a .tgq artifact as an EnsembleIR with .predict(X)."""
    from .ir import Tree

    with open(path, "rb") as fh:
        magic = fh.read(4)
        if magic != _MAGIC:
            raise ValueError("not a .tgq file")
        data = np.load(io.BytesIO(lzma.decompress(fh.read())))
    objective = bytes(data["objective"]).decode()
    decision = bytes(data["decision"]).decode()
    weights = data["tree_weights"].astype(np.float64)
    codebook = data["codebook"].astype(np.float64) if "codebook" in data else None
    trees = []
    t = 0
    while f"f{t}" in data:
        feature = data[f"f{t}"].astype(np.int32)
        left = data[f"l{t}"].astype(np.int32)
        right = data[f"r{t}"].astype(np.int32)
        n = len(feature)
        default_left = np.unpackbits(data[f"d{t}"])[:n].astype(bool)
        value = np.zeros((n, 1))
        tree = Tree(feature, data[f"t{t}"].astype(np.float64), left, right,
                    default_left, value)
        lid = tree.leaf_ids()
        if codebook is not None and f"c{t}" in data:
            value[lid, 0] = codebook[data[f"c{t}"]]
        else:
            value[lid, 0] = data[f"v{t}"].astype(np.float64)
        trees.append(tree)
        t += 1
    return EnsembleIR(trees=trees, base_margin=float(data["base_margin"][0]),
                      objective=objective, decision=decision, tree_weights=weights)
