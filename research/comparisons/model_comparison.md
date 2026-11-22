# Recorded model comparison

Audit date: 2026-11-22. These are previously recorded results; no model was trained or evaluated during this audit.

| Experiment | Resolution divisor | Iterations | PSNR (dB) | SSIM | LPIPS | Protocol |
| --- | --- | --- | --- | --- | --- | --- |
| b0-kaggle-c55f001d26 | 1 | 30000 | 20.913033 | 0.695186 | 0.269434 | rounded_uint8_png |
| ignatius-legacy-b0-low-memory_20261115T141109Z_8ed1c6 | 4 | 30000 | 23.375393 | 0.839926 | 0.115017 | rounded_uint8_png |
| ignatius-legacy-b0-research-quality_20261115T222504Z_9ddaaa | 1 | 7400 | NOT RECORDED | NOT RECORDED | NOT RECORDED | NOT RECORDED |
| ignatius-legacy-b0-smoke_20261115T133608Z_4ef4c4 | 4 | 0 | NOT RECORDED | NOT RECORDED | NOT RECORDED | NOT RECORDED |
| ignatius-legacy-b0-smoke_20261115T134942Z_346824 | 4 | 1200 | 15.026311 | 0.485911 | 0.417408 | rounded_uint8_png |
| kaggle-pgsr-geometry-smoke | 8 | 300 | 9.896437 | 0.225452 | 0.550352 | float_render_diagnostic |
| local-smoke-v1 | 8 | 300 | 9.826858 | 0.222708 | 0.556722 | rounded_uint8_png |
| pgsr-kaggle-c269454133 | 1 | 21298 | NOT RECORDED | NOT RECORDED | NOT RECORDED | NOT RECORDED |
| B1_2DGS_NOT_TRAINED | N/A | N/A | NOT RECORDED | NOT RECORDED | NOT RECORDED | NOT TRAINED |

There is no completed, matched-protocol three-method comparison. B1 2DGS was not trained. B2 research stopped at21298/30000 and has no recorded final RGB evaluation. B0 full-resolution and low-memory use different resolutions; their scores must not be interpreted as a controlled superiority result. Development smoke scores are diagnostics.

PGSR total loss includes weighted geometric terms; B0 total loss is RGB-only. Never rank their total objectives numerically. Empty fields mean unavailable, not zero. Test and periodic validation results are separate. See [pairwise audit](comparability_audit.json) and [registry](../metadata/experiment_registry.json).

Runtime columns are summed maximum elapsed values from observed log segments, hence lower bounds. GPU models and iteration windows differ. Reported last-segment durations and memory scopes remain in the registry. No FPS, system-RAM peak, dense geometry error or reconstructed mesh quality was recovered.
