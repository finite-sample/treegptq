"""Adapters between native fitted models and the IR.

Contract per framework:
  to_ir(model, X_ref)   -> EnsembleIR   (X_ref: small sample used to pin the
                                          base margin empirically and verify
                                          the conversion end to end)
  patch_back(model, ir) -> native model of the same class with the IR's leaf
                           values written into it. Structure is never touched.

Supported in v0.1:
  xgboost.Booster / xgboost.XGBRegressor / xgboost.XGBClassifier (binary)
  sklearn.ensemble.RandomForestRegressor
  sklearn.ensemble.GradientBoostingRegressor
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from .ir import EnsembleIR, Tree

_ATOL = 1e-4


def _pin_base_margin(ir: EnsembleIR, native_margin: np.ndarray, X_ref: np.ndarray) -> None:
    """Set base_margin so IR margin matches the native model on X_ref exactly."""
    ir.base_margin = 0.0
    diff = native_margin - ir.margin(X_ref)
    base = float(np.mean(diff))
    if np.std(diff) > _ATOL:
        raise RuntimeError(
            f"IR conversion mismatch: residual base margin is not constant "
            f"(std={np.std(diff):.2e}). Unsupported model configuration."
        )
    ir.base_margin = base


# ---------------------------------------------------------------- xgboost

def _xgb_booster(model: Any):
    import xgboost as xgb

    if isinstance(model, xgb.Booster):
        return model
    return model.get_booster()


def _xgb_to_ir(model: Any, X_ref: np.ndarray) -> EnsembleIR:
    import xgboost as xgb

    booster = _xgb_booster(model)
    cfg = json.loads(bytearray(booster.save_raw("json")))
    learner = cfg["learner"]
    objective_name = learner["objective"]["name"]
    if objective_name == "reg:squarederror":
        objective = "reg"
    elif objective_name == "binary:logistic":
        objective = "binary_logistic"
    else:
        raise NotImplementedError(f"xgboost objective {objective_name!r} not supported in v0.1")
    if int(learner["learner_model_param"].get("num_class", "0")) > 1:
        raise NotImplementedError("multiclass xgboost not supported in v0.1")

    trees = []
    for tr in learner["gradient_booster"]["model"]["trees"]:
        left = np.asarray(tr["left_children"], dtype=np.int32)
        right = np.asarray(tr["right_children"], dtype=np.int32)
        feature = np.asarray(tr["split_indices"], dtype=np.int32)
        threshold = np.asarray(tr["split_conditions"], dtype=np.float64)
        default_left = np.asarray(tr["default_left"], dtype=bool)
        is_leaf = left < 0
        value = np.where(is_leaf, threshold, 0.0)[:, None]  # leaf value lives in split_conditions
        feature = np.where(is_leaf, -1, feature)
        trees.append(Tree(feature, np.where(is_leaf, 0.0, threshold), left, right,
                          default_left, value))

    ir = EnsembleIR(trees=trees, base_margin=0.0, objective=objective, decision="lt",
                    meta={"framework": "xgboost"})
    native_margin = booster.predict(xgb.DMatrix(X_ref), output_margin=True).astype(np.float64)
    _pin_base_margin(ir, native_margin, np.asarray(X_ref, dtype=np.float64))
    return ir


def _xgb_patch_back(model: Any, ir: EnsembleIR) -> Any:
    import xgboost as xgb

    booster = _xgb_booster(model)
    cfg = json.loads(bytearray(booster.save_raw("json")))
    jtrees = cfg["learner"]["gradient_booster"]["model"]["trees"]
    if len(jtrees) != len(ir.trees):
        raise ValueError("tree count mismatch between model and IR")
    for jt, tree in zip(jtrees, ir.trees):
        left = np.asarray(jt["left_children"])
        sc = np.asarray(jt["split_conditions"], dtype=np.float32)
        bw = np.asarray(jt["base_weights"], dtype=np.float32)
        leaves = left < 0
        vals = tree.value[:, 0].astype(np.float32)
        sc[leaves] = vals[leaves]
        bw[leaves] = vals[leaves]
        jt["split_conditions"] = [float(v) for v in sc]
        jt["base_weights"] = [float(v) for v in bw]
    patched = xgb.Booster()
    patched.load_model(bytearray(json.dumps(cfg), "utf-8"))
    if not isinstance(model, xgb.Booster):
        import copy

        wrapper = copy.copy(model)
        wrapper._Booster = patched
        return wrapper
    return patched


# ---------------------------------------------------------------- sklearn

def _skl_tree_to_ir_tree(t, scale: float = 1.0) -> Tree:
    feature = t.feature.astype(np.int32).copy()
    threshold = t.threshold.astype(np.float64).copy()
    left = t.children_left.astype(np.int32).copy()
    right = t.children_right.astype(np.int32).copy()
    is_leaf = left < 0
    feature[is_leaf] = -1
    threshold[is_leaf] = 0.0
    value = (t.value[:, 0, 0].astype(np.float64) * scale)[:, None]
    return Tree(feature, threshold, left, right,
                np.ones_like(is_leaf, dtype=bool), value)


def _rf_to_ir(model: Any, X_ref: np.ndarray) -> EnsembleIR:
    trees = [_skl_tree_to_ir_tree(est.tree_) for est in model.estimators_]
    w = np.full(len(trees), 1.0 / len(trees))
    ir = EnsembleIR(trees=trees, base_margin=0.0, objective="reg", decision="le",
                    tree_weights=w, meta={"framework": "sklearn_rf"})
    _pin_base_margin(ir, model.predict(X_ref).astype(np.float64),
                     np.asarray(X_ref, dtype=np.float64))
    return ir


def _rf_patch_back(model: Any, ir: EnsembleIR) -> Any:
    import copy

    patched = copy.deepcopy(model)
    for est, tree in zip(patched.estimators_, ir.trees):
        leaves = est.tree_.children_left < 0
        est.tree_.value[leaves, 0, 0] = tree.value[leaves, 0]
    return patched


def _gbr_to_ir(model: Any, X_ref: np.ndarray) -> EnsembleIR:
    lr = float(model.learning_rate)
    trees = [_skl_tree_to_ir_tree(model.estimators_[i, 0].tree_)
             for i in range(model.n_estimators_)]
    w = np.full(len(trees), lr)
    ir = EnsembleIR(trees=trees, base_margin=0.0, objective="reg", decision="le",
                    tree_weights=w, meta={"framework": "sklearn_gbr"})
    _pin_base_margin(ir, model.predict(X_ref).astype(np.float64),
                     np.asarray(X_ref, dtype=np.float64))
    return ir


def _gbr_patch_back(model: Any, ir: EnsembleIR) -> Any:
    import copy

    patched = copy.deepcopy(model)
    for i in range(patched.n_estimators_):
        t = patched.estimators_[i, 0].tree_
        leaves = t.children_left < 0
        t.value[leaves, 0, 0] = ir.trees[i].value[leaves, 0]
    return patched


# ---------------------------------------------------------------- dispatch

def _kind(model: Any) -> str:
    name = type(model).__name__
    mod = type(model).__module__
    if mod.startswith("xgboost") or name == "Booster":
        return "xgboost"
    if name == "RandomForestRegressor":
        return "sklearn_rf"
    if name == "GradientBoostingRegressor":
        return "sklearn_gbr"
    raise NotImplementedError(
        f"{mod}.{name} not supported in v0.1. Supported: xgboost (reg, binary), "
        f"RandomForestRegressor, GradientBoostingRegressor."
    )


def to_ir(model: Any, X_ref: np.ndarray) -> EnsembleIR:
    return {"xgboost": _xgb_to_ir, "sklearn_rf": _rf_to_ir,
            "sklearn_gbr": _gbr_to_ir}[_kind(model)](model, X_ref)


def patch_back(model: Any, ir: EnsembleIR) -> Any:
    return {"xgboost": _xgb_patch_back, "sklearn_rf": _rf_patch_back,
            "sklearn_gbr": _gbr_patch_back}[_kind(model)](model, ir)
