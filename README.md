# treegptq

Post-training quantization for fitted tree ensembles. Bring your trained
XGBoost or scikit-learn model, get back the same model with 2-4 bit leaf
values, a compact `.tgq` artifact, and a report stating exactly how much
space you saved and how much accuracy it cost.

The method adapts GPTQ's error-compensation idea from LLM quantization to
additive tree models: trees are quantized in ensemble order, and each tree's
leaf values are adjusted to cancel the accumulated quantization error of its
predecessors before being quantized themselves. No retraining. A calibration
sample and seconds of CPU.

## Quickstart

```python
import treegptq as tq

# model: a fitted xgboost.Booster / XGBRegressor / XGBClassifier (binary),
#        sklearn RandomForestRegressor, or GradientBoostingRegressor
qmodel, report = tq.quantize(model, X_cal, y_cal, bits=3,
                             X_eval=X_test, y_eval=y_test)
print(report.summary())
# gptq@3b: 4.9x smaller (1,204,882 -> 244,655 bytes) | rmse 0.4582 -> 0.4641 (+1.29%)

qmodel.predict(X)        # same class as your model; your pipeline unchanged

tq.save(qmodel, "model.tgq", X_ref=X_cal)   # compact artifact
ir = tq.load("model.tgq"); ir.predict(X)    # numpy-only inference
```

`sweep()` produces the full method-by-bits frontier on held-out data so the
size/accuracy trade is a table, not a guess.

## Methods

`rtn`, `rtn_global` (uniform grids), `kmeans`, `wkmeans` (global codebooks,
the latter weighted by calibration traffic times hessian mass), `gptq`
(sequential error compensation, stability-clipped), `gptq_cb` (compensation
onto the weighted codebook), `gptq_alloc` (compensation plus per-tree bit
allocation). On our benchmarks compensation cuts quantization-induced loss
3-6x versus round-to-nearest at 2-3 bits.

## Design

- One IR, thin adapters. Vector-valued leaves in the IR from day one so
  multiclass lands without a rewrite.
- Patch-back, not a new runtime: `quantize()` returns your model's own class
  with quantized values written into it. Inference code, latency, and serving
  paths are untouched.
- The report is the product: bytes before/after (both post-compression) and
  the task metric before/after, always.

## Supported (v0.1)

xgboost `reg:squarederror` and `binary:logistic`; sklearn
`RandomForestRegressor` and `GradientBoostingRegressor`. Roadmap: LightGBM,
HistGradientBoosting, multiclass, threshold quantization, pruning and
distillation baselines for matched-byte comparisons, CatBoost, EBM.

## License

MIT
