# ONNX inputs and outputs checks

`src/data/docs/onnx-io/<family>.json` feeds the `<onnx-io />` block on each model
page. Every entry is backed by `check_<family>.py`, which exports the smallest
checkpoint with LibreYOLO, builds the input and decodes the outputs with numpy,
OpenCV and onnxruntime from the JSON alone, and compares the result with
`LibreYOLO(<file>.pt).predict`. An entry ships only when its check passes.

Rerun after every LibreYOLO release, in an environment with
`pip install "libreyolo[onnx,rfdetr]==<version>"`:

```bash
for f in scripts/onnx-io/check_*.py; do
  OMP_NUM_THREADS=2 python "$f" /tmp/onnx-io-work || echo "FAILED: $f"
done
```

Then bump `verified_with` in each JSON. Checkpoints download into `./weights/`;
delete that folder afterwards. On Apple Silicon an export occasionally traces
differently on MPS; rerun a single failing check before trusting a failure.

Adding a family: copy `yolo9.json` and `check_yolo9.py`, build the input with
`_common.preprocess`, then run
`node scripts/onnx-io/insert-section.mjs <page-slug>` and
`node scripts/translation/sync-check.mjs --stamp`.
