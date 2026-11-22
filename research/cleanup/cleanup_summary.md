# Cleanup summary

Audit date: 2026-11-22. Status: SUCCESS.

- Pre-cleanup logical bytes: 52500485363 across 18926 files.
- Final logical bytes at verification scan: 52508197406 across 18811 files. Generated reports/figures and metadata growth are included; this is not a measure of deletion savings.
- Files removed: 127; exact logical bytes removed: 953633.
- Checkpoints/PLYs/meshes/datasets/COLMAP/configs/original logs removed: 0.
- Checkpoint/weight files retained and reopened: 58.
- Experiment records: 8; source loss observations retained: 907.
- Full-state structural resume candidates: 6. CUDA resumption and canonical layout compatibility remain separate gates.
- No dataset, checkpoint or experiment directory was moved. No recursive directory deletion was used.

Potentially valuable duplicate artifacts remain in [approval_required.csv](approval_required.csv). Their canonical hashes and exact/basename/manifest dependencies are in [dependency_audit.csv](dependency_audit.csv). Equality does not retire a sealed bundle or its referenced paths. All deletion candidates require explicit approval except the enumerated cache files already removed.

The pre-cleanup and remaining inventories include managed environments, external code and Git internals, which were preserved. Logical file lengths are reported separately from the observed volume free-space delta in [cleanup_manifest.json](cleanup_manifest.json); concurrent writes and allocation units can make those quantities differ. Audit-generated files may be newer than the final snapshot.

[Final verification](../metadata/final_verification.json) distinguishes verified file/state accessibility from workflows not executed. B2's non-native canonical layout is documented in [reproducibility](../../docs/EXPERIMENT_REPRODUCIBILITY.md). No training, inference or mesh benchmark was run during this audit.
