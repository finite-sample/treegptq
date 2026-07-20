import numpy as np
import pytest
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

import treegptq as tq
from treegptq import adapters

X, y = make_friedman1(n_samples=3000, noise=1.0, random_state=0)
Xc, yc = X[:2000], y[:2000]
Xt, yt = X[2000:], y[2000:]


def _xgb_reg():
    import xgboost as xgb
    d = xgb.DMatrix(Xc, label=yc)
    return xgb.train({"max_depth": 4, "eta": 0.3, "seed": 0,
                      "objective": "reg:squarederror"}, d, 50)


def _xgb_clf():
    import xgboost as xgb
    yb = (yc > np.median(yc)).astype(int)
    d = xgb.DMatrix(Xc, label=yb)
    return xgb.train({"max_depth": 4, "eta": 0.3, "seed": 0,
                      "objective": "binary:logistic"}, d, 50)


def test_xgb_ir_matches_native():
    import xgboost as xgb
    bst = _xgb_reg()
    ir = adapters.to_ir(bst, Xc[:256])
    np.testing.assert_allclose(ir.predict(Xt), bst.predict(xgb.DMatrix(Xt)), atol=1e-4)


def test_xgb_logistic_ir_matches_native():
    import xgboost as xgb
    bst = _xgb_clf()
    ir = adapters.to_ir(bst, Xc[:256])
    np.testing.assert_allclose(ir.predict(Xt), bst.predict(xgb.DMatrix(Xt)), atol=1e-4)


def test_rf_ir_matches_native():
    rf = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=0).fit(Xc, yc)
    ir = adapters.to_ir(rf, Xc[:256])
    np.testing.assert_allclose(ir.predict(Xt), rf.predict(Xt), atol=1e-8)


def test_gbr_ir_matches_native():
    gb = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=0).fit(Xc, yc)
    ir = adapters.to_ir(gb, Xc[:256])
    np.testing.assert_allclose(ir.predict(Xt), gb.predict(Xt), atol=1e-8)


@pytest.mark.parametrize("kind", ["xgb", "rf"])
def test_patch_back_matches_ir(kind):
    import xgboost as xgb
    model = _xgb_reg() if kind == "xgb" else RandomForestRegressor(
        n_estimators=30, max_depth=6, random_state=0).fit(Xc, yc)
    qmodel, _ = tq.quantize(model, Xc, yc, bits=3, method="gptq")
    ir = adapters.to_ir(model, Xc[:256])
    from treegptq.quant import quantize_leaves
    qir, _ = quantize_leaves(ir, Xc, bits=3, method="gptq")
    native = (qmodel.predict(xgb.DMatrix(Xt)) if isinstance(qmodel, xgb.Booster)
              else qmodel.predict(Xt))
    np.testing.assert_allclose(native, qir.predict(Xt), atol=1e-4)


def test_high_bits_identity():
    rf = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=0).fit(Xc, yc)
    qmodel, _ = tq.quantize(rf, Xc, yc, bits=8, method="rtn")
    assert np.sqrt(np.mean((qmodel.predict(Xt) - rf.predict(Xt))**2)) < 0.02


def test_save_load_roundtrip(tmp_path):
    bst = _xgb_reg()
    ir = adapters.to_ir(bst, Xc[:256])
    from treegptq.quant import quantize_leaves
    qir, info = quantize_leaves(ir, Xc, bits=3, method="gptq_cb")
    p = str(tmp_path / "m.tgq")
    tq.save(qir, p, info=info)
    loaded = tq.load(p)
    np.testing.assert_allclose(loaded.predict(Xt), qir.predict(Xt), atol=1e-5)
