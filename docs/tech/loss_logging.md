# Loss logging contract

Future B0 Original3DGS and B2 PGSR runs write one append-only `training.jsonl` file per run. The normalized `losses` object uses schema `gaussian-loss-v1`; legacy top-level fields remain temporarily available for existing consumers. This avoids duplicate histories while making each sampled observation directly usable by the research audit.

## Common fields

Each logged row records `iteration`, sampled image identity, `loss` and `losses.total`, Gaussian count, elapsed segment time, optimizer progress, CUDA memory telemetry and method-specific diagnostics. `losses.raw` contains unweighted terms, `losses.coefficients` contains the configured multipliers, and `losses.weighted` contains the actual additive contributions. `losses.activation` distinguishes a scheduled-inactive zero from a measured active zero.

The same total, raw, and weighted loss values are printed directly to the live training console at each configured `log_interval`. The append-only JSONL remains the authoritative machine-readable evidence if console output is truncated by a remote session.

Logging is sampled at the configured `log_interval`; it is not a per-iteration history unless that interval is one. Missing iterations and missing components remain unavailable and must never be filled with zero or interpolation.

## B0 Original3DGS

The raw terms are `l1` and `ssim_dissimilarity`, where the latter is `1 - SSIM`. Their coefficients are `1 - lambda_dssim` and `lambda_dssim`. The sum of the two weighted contributions equals `losses.total` up to floating-point rounding.

## B2 PGSR

The additive objective terms are `rgb`, `flatten`, `single`, `multi_geo` and `multi_ncc`. The `photometric` block additionally records:

- `l1_uncompensated`: L1 from canonical rendered RGB.
- `l1_appearance`: L1 from the appearance-compensated branch when that branch exists.
- `l1_selected`: the L1 branch selected by the native exposure gate.
- `ssim_dissimilarity`: `1 - SSIM` from canonical rendered RGB.
- both photometric coefficients and `exposure_gate_used`.

The exposure model changes the selected L1 branch; it is not a separate additive exposure loss. Geometry coverage counts are stored beside activation flags so a zero geometry loss is interpretable.

## Historical and comparison policy

Historical logs are immutable. The audit parser reads `gaussian-loss-v1` when present and falls back to legacy fields. It therefore leaves old PGSR standalone L1/SSIM values unavailable instead of reconstructing them. Total loss values are only comparable within an identical objective and configuration; B0 and PGSR totals must not be ranked against each other.

The canonical reporting outputs remain under `research/metrics/<experiment_id>/`: `loss_history.csv`, `loss_definitions.json`, and `loss_summary.json`. Existing reports that already contain the required detail do not need regeneration solely for this schema change.

Print every experiment summary directly:

```powershell
python scripts/research/print_losses.py
```

Print all recovered iteration rows for selected experiments and preserve the exact console text for a report:

```powershell
python scripts/research/print_losses.py `
  --experiment b0-kaggle-c55f001d26 `
  --experiment pgsr-kaggle-c269454133 `
  --history `
  --output research/reports/loss_process_output.txt
```

This command is read-only with respect to training artifacts. It prints `N/A` for unavailable values and never reconstructs historical PGSR L1/SSIM terms.
