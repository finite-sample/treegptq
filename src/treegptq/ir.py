"""Framework-neutral intermediate representation for tree ensembles.

Design notes:
- Leaves are vector-valued (shape (n_nodes, n_out)) from day one so multiclass
  and multi-output models slot in without an IR rewrite. v0.1 exercises n_out=1.
- Prediction is margin-space: margin(x) = base + sum_t w_t * leaf_t(x).
  `objective` records the link for metric computation ("reg" identity,
  "binary_logistic" sigmoid).
- `decision` records the split comparator: "lt" (xgboost: go left iff x < thr)
  or "le" (sklearn: go left iff x <= thr). Missing values follow default_left.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Tree:
    feature: np.ndarray        # (n_nodes,) int32, -1 at leaves
    threshold: np.ndarray      # (n_nodes,) float64
    left: np.ndarray           # (n_nodes,) int32, -1 at leaves
    right: np.ndarray          # (n_nodes,) int32, -1 at leaves
    default_left: np.ndarray   # (n_nodes,) bool
    value: np.ndarray          # (n_nodes, n_out) float64; meaningful at leaves

    @property
    def is_leaf(self) -> np.ndarray:
        return self.left < 0

    @property
    def n_leaves(self) -> int:
        return int(self.is_leaf.sum())

    def leaf_ids(self) -> np.ndarray:
        """Dense re-indexing: node index of each leaf, sorted."""
        return np.flatnonzero(self.is_leaf)


@dataclass
class EnsembleIR:
    trees: list[Tree]
    base_margin: float
    objective: str                       # "reg" | "binary_logistic"
    decision: str                        # "lt" | "le"
    n_out: int = 1
    tree_weights: np.ndarray | None = None   # (n_trees,), defaults to 1.0
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tree_weights is None:
            self.tree_weights = np.ones(len(self.trees))

    # ---------------- traversal ----------------

    def _descend(self, tree: Tree, X: np.ndarray) -> np.ndarray:
        """Vectorized routing: node index of the leaf each row reaches."""
        n = X.shape[0]
        idx = np.zeros(n, dtype=np.int64)
        active = ~tree.is_leaf[idx]
        while active.any():
            cur = idx[active]
            f = tree.feature[cur]
            thr = tree.threshold[cur]
            x = X[active, f].astype(np.float32)  # both frameworks route on float32 features
            nan = np.isnan(x)
            if self.decision == "lt":            # xgboost: float32 vs float32
                go_left = x < thr.astype(np.float32)
            else:                                # sklearn: float32 vs float64
                go_left = x <= thr
            go_left = np.where(nan, tree.default_left[cur], go_left)
            idx[active] = np.where(go_left, tree.left[cur], tree.right[cur])
            active = ~tree.is_leaf[idx]
        return idx

    def leaf_assignment(self, X: np.ndarray) -> np.ndarray:
        """(n_samples, n_trees) node index of the reached leaf per tree."""
        X = np.asarray(X, dtype=np.float64)
        out = np.empty((X.shape[0], len(self.trees)), dtype=np.int64)
        for t, tree in enumerate(self.trees):
            out[:, t] = self._descend(tree, X)
        return out

    def margin(self, X: np.ndarray, assign: np.ndarray | None = None) -> np.ndarray:
        """Margin-space prediction, shape (n_samples,) for n_out == 1."""
        if assign is None:
            assign = self.leaf_assignment(X)
        n = assign.shape[0]
        out = np.full((n, self.n_out), self.base_margin, dtype=np.float64)
        for t, tree in enumerate(self.trees):
            out += self.tree_weights[t] * tree.value[assign[:, t]]
        return out[:, 0] if self.n_out == 1 else out

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Output-space prediction (probability for binary_logistic)."""
        m = self.margin(X)
        if self.objective == "binary_logistic":
            return 1.0 / (1.0 + np.exp(-m))
        return m

    # ---------------- leaf-value plumbing ----------------

    def get_leaf_values(self) -> list[np.ndarray]:
        """Per tree: (n_leaves, n_out) leaf values in leaf_ids() order."""
        return [t.value[t.leaf_ids()].copy() for t in self.trees]

    def set_leaf_values(self, values: list[np.ndarray]) -> EnsembleIR:
        for tree, v in zip(self.trees, values):
            tree.value[tree.leaf_ids()] = np.asarray(v).reshape(tree.n_leaves, self.n_out)
        return self

    def copy(self) -> EnsembleIR:
        return EnsembleIR(
            trees=[Tree(t.feature.copy(), t.threshold.copy(), t.left.copy(),
                        t.right.copy(), t.default_left.copy(), t.value.copy())
                   for t in self.trees],
            base_margin=self.base_margin,
            objective=self.objective,
            decision=self.decision,
            n_out=self.n_out,
            tree_weights=None if self.tree_weights is None else self.tree_weights.copy(),
            meta=dict(self.meta),
        )

    @property
    def n_leaves_total(self) -> int:
        return sum(t.n_leaves for t in self.trees)
