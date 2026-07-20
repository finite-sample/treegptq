import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.ensemble import RandomForestRegressor

import treegptq as tq
from treegptq import adapters
from treegptq.quant import METHODS, quantize_leaves

X, y = make_friedman1(n_samples=4000, noise=1.0, random_state=1)
Xc, yc, Xt, yt = X[:3000], y[:3000], X[3000:], y[3000:]


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def test_all_methods_run_and_bound():
    rf = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=0).fit(Xc, yc)
    ir = adapters.to_ir(rf, Xc[:256])
    base = _rmse(ir.predict(Xt), yt)
    for m in METHODS:
        qir, info = quantize_leaves(ir, Xc, bits=3, method=m)
        r = _rmse(qir.predict(Xt), yt)
        assert np.isfinite(r) and r < 3 * base, (m, r, base)


def test_gptq_beats_rtn_at_2bits():
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=0).fit(Xc, yc)
    ir = adapters.to_ir(rf, Xc[:256])
    q_rtn, _ = quantize_leaves(ir, Xc, bits=2, method="rtn")
    q_gptq, _ = quantize_leaves(ir, Xc, bits=2, method="gptq")
    assert _rmse(q_gptq.predict(Xt), yt) <= _rmse(q_rtn.predict(Xt), yt) * 1.02


def test_sweep_reports():
    rf = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=0).fit(Xc, yc)
    reports = tq.sweep(rf, Xc, yc, Xt, yt, bits=[2, 4], methods=["rtn", "gptq"])
    assert len(reports) == 4
    for r in reports:
        assert r.compression > 1.0
        assert "rmse" in r.metrics_after
