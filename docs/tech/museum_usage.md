# Museum reconstruction

The input is `datasets/tanks_and_temples/museum/Museum.mp4`. Model runs and SfM reports are isolated under `outputs/tanks_and_temples/Museum/`. The raw video stays in place. Museum belongs to the advanced test set. No qualified local GT geometry, depth or normal maps were found. [Tanks and Temples downloads](https://tanksandtemples.org/download/).

## Run strict SfM and prepare the shared scene

Run from the repository root in PowerShell with CUDA Torch and FFmpeg. Point `COLMAP_BIN` at a CUDA/CASPAR-enabled build that passes the repository's synthetic BA probe. The verified alternate Windows release archive is `COLMAP-4.2.1-windows-2022-CUDA-Caspar.zip`, SHA-256 `7fa0a9f0e647ddb9c7fc049842877dab1c380664e9adbee565aafc0aa9fbada5`. The wrapper disables structure-less registration when the CLI offers that toggle. The runtime gate still rejects Ceres and fallback log entries. [Pinned Windows build](https://github.com/lyehe/build_gpu_colmap/releases/tag/v4.2.1). [Upstream registration options](https://github.com/colmap/colmap/blob/4.2.1/src/colmap/controllers/incremental_pipeline.cc).

```powershell
$env:COLMAP_BIN = (Resolve-Path .tools/colmap-caspar-4.2.1-windows/bin/colmap.exe).Path
$env:FFMPEG_BIN = (Resolve-Path .tools/ffmpeg/ffmpeg-9.0.2-essentials_build/bin/ffmpeg.exe).Path
$env:FFPROBE_BIN = (Resolve-Path .tools/ffmpeg/ffmpeg-9.0.2-essentials_build/bin/ffprobe.exe).Path

# Fresh workspace with no existing paths listed by this configuration.
python -B -m src.pipeline --config configs/sfm_museum_strict.yaml --stage all

# For the recorded retry, extraction already existed. Use select, then colmap --resume.
python -B -m scripts.kaggle.museum_handoff scene --sfm-config configs/sfm_museum_strict.yaml `
  --destination outputs/tanks_and_temples/Museum/kaggle_handoff/scene_v1
python -B -m scripts.kaggle.museum_handoff pack-scene `
  outputs/tanks_and_temples/Museum/kaggle_handoff/scene_v1 `
  --destination outputs/tanks_and_temples/Museum/kaggle_handoff/scene_transport_v1
```

The strict profile samples candidates at 2 fps, caps width at 1280, targets 160 keyframes and caps the selection at 240. The maximum allowed time gap stays 8 seconds. The initial 96-target profile failed that check. The coverage retry passed selection but the mapper invoked Ceres during structure-less registration. The strict retry retained the coverage settings and the original candidate frames. Its separate output keeps both earlier attempts available.

The scene command requires successful GPU-SfM evidence and a matching source-video SHA-256. If a completed CASPAR solve fails solely on repeated image IDs in tracks, the Museum config allows a separate derived export that drops those whole tracks and revalidates all remaining geometry. The command records original model and mapper-log hashes. An earlier mapping that invoked Ceres cannot enter this path.

Phase01 prepares undistorted frames and train-only Gaussian initialization. Its declared policy drops whole tracks with observations outside the transformed image. It records world-to-camera Hamilton WXYZ and a frozen interleaved split with transductive SfM poses. Scale is arbitrary. Further qualification errors stop the handoff.

## Local profiles and smoke runs

Bind a profile from `configs/training/museum/` to the exact generated dataset directory. Use the existing isolated B0 and B2 interpreters and the corresponding profile name. Preserve the same dataset ID and image resolution for research comparisons. Smoke and low-memory profiles change the optimization workload and have separate labels.

```powershell
python -B -m scripts.kaggle.museum_handoff profile --dataset <prepared-museum-directory> `
  --method B0 --profile smoke --destination <new-resolved-b0-yaml>
.\.tools\b0_env\Scripts\python.exe -B -m src.gaussian_splatting.runner `
  --config <resolved-b0-yaml> --stop-after 550
.\.tools\b0_env\Scripts\python.exe -B -m src.gaussian_splatting.runner `
  --config <resolved-b0-yaml> --resume <b0-run>
.\.tools\b0_env\Scripts\python.exe -B -m src.gaussian_splatting.runner `
  --config <resolved-b0-yaml> --run <b0-run> --stage evaluate

python -B -m scripts.kaggle.museum_handoff profile --dataset <prepared-museum-directory> `
  --method B2 --profile smoke --destination <new-resolved-pgsr-yaml>
.\.tools\pgsr_env\Scripts\python.exe -B -m src.pgsr.runner `
  --config <resolved-pgsr-yaml> --run <new-pgsr-run> --stop-after 80
.\.tools\pgsr_env\Scripts\python.exe -B -m src.pgsr.runner `
  --config <resolved-pgsr-yaml> --run <pgsr-run> --resume
.\.tools\pgsr_env\Scripts\python.exe -B -m src.pgsr.runner `
  --config <resolved-pgsr-yaml> --run <pgsr-run> --stage evaluate
```

## Kaggle inputs and results

1. Generate `museum_source.bundle` with `python -B -m scripts.kaggle.museum_handoff source --destination <new-source-folder>`. Attach it as a private dataset. Set `SOURCE_SHA256` using `source_manifest.json`.
2. Attach the **same** `museum_scene.bundle` to [B0](../../notebooks/kaggle/phase02_resume_3dgs.ipynb) and [PGSR](../../notebooks/kaggle/phase03_pgsr_run_all.ipynb). Set `SCENE_ARCHIVE_SHA256` and `SCENE_RECEIPT_SHA256` from `scene_manifest.json`. An extracted copy can use `SCENE_INPUT` with the receipt hash. A fresh SfM session instead requires the raw MP4 and a verified Linux CUDA/CASPAR build delimited by the same runtime gate.
3. Enable one usable CUDA device and Internet, then run one method at a time. Research profiles target 30,000 iterations. Save the outputs as a version. Resume only from the same method and profile with `PREVIOUS_OUTPUT` and `PREVIOUS_CHECKPOINT_SHA256`. The full-state native restore also checks code, configuration, scene and backend identity.
4. Attach the saved Museum PGSR result to [the mesh notebook](../../pgsr_mesh_kaggle.ipynb). Set `MUSEUM_OUTPUT` to the saved directory and provide the printed receipt and checkpoint hashes. Its CUDA TSDF variant uses train views and exports positive axial-Z depth, camera-frame unit normals, masks, K/Tcw and a triangle PLY. B0 exports a Gaussian PLY and RGB renders.
5. Import a saved training output with `python -B -m scripts.kaggle.museum_handoff import-run <downloaded-output-directory>`. The same importer is available through both historical import scripts with `--museum`. Refresh the Museum experiment registry with `python -B scripts/research/analyze.py --museum`.

The verified shared inputs are also available privately on Kaggle. The published B0 and PGSR notebooks pin both package hashes and use the same frozen scene. [Source package](https://www.kaggle.com/datasets/vthkhnhhuyn/museum-reconstruction-source-v3) · [prepared scene](https://www.kaggle.com/datasets/vthkhnhhuyn/museum-prepared-video-v1) · [B0 notebook](https://www.kaggle.com/code/vthkhnhhuyn/museum-original-3dgs-v1) · [PGSR notebook](https://www.kaggle.com/code/vthkhnhhuyn/museum-pgsr-v1).

`training.jsonl` preserves observed loss components. The Kaggle export invokes `scripts.kaggle.loss_report` to verify weighted sums and produce a sampled CSV and PNG without interpolating missing rows. Run it locally with `python -B -m scripts.kaggle.loss_report <training.jsonl> <new-loss-report-folder>`. PGSR publishes separate float-render diagnostics and PNG scores. Compare matching PNG scores only when the profiles and splits qualify. Geometry scores require an approved Museum benchmark submission or a qualified local reference.
