# Weights

`MODEL_CARD.json` and `SHA256SUMS` document the EMA runs behind [`results/`](../results/). Train with the scripts under `scripts/`, then pass `--ckpt` / `--init` to demos and evals.

Reprint headline tables on CPU:

```bash
python scripts/print_result_tables.py
```

Checkpoint format used by the training/eval scripts (`loopworld_infer_ema_v1`):

```python
ck = torch.load("path/to/your_ema.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ck["ema"])
cfg = ck["config"]
```
