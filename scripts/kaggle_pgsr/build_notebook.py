"""Generate the hash-pinned twelve-section Kaggle Run All notebook from a sealed capsule."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.common.io import read_json,write_json


def build(transport,output):
    lock=read_json(transport)
    cells=[]
    def section(title,description,code):
        cells.append({"cell_type":"markdown","metadata":{},"source":f"## {title}\n\n{description}".splitlines(True)})
        cells.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":code.strip().splitlines(True)})
    section("1. PGSR research run","Official PGSR, fixed Ignatius 197/33/33 split. New PGSR optimization from train-only SfM; geometry GT is never loaded. Run All performs CUDA verification, real-data smoke/resume, full training, held-out RGB evaluation and plane-depth/normal export. A partial session produces a resumable full-state recovery bundle.",'''
import os, sys, json, hashlib, zipfile, shutil, subprocess
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='0'
WORK=Path('/tmp/phase03-pgsr-v1')
PROJECT=WORK/'project'
OUTPUT=Path('/kaggle/working/phase03_results')
# For a later session, attach the previous notebook output and set this to its recovery/ folder.
PREVIOUS=None
''')
    section("2. GPU preflight","The separate environment and actual native forward/backward tests establish CUDA execution. Only cuda:0 is used even if Kaggle exposes two GPUs.",'''
subprocess.run(['nvidia-smi'],check=True)
assert not OUTPUT.exists(), 'Existing output: use a new session; never overwrite a published run'
''')
    section("3. Verify and unpack source","The notebook pins the complete private source capsule hash before importing any bundled project code. All archive destinations stay within the new workspace.",f'''
SOURCE_SHA={lock['archive_sha256']!r}
SOURCE_MANIFEST_SHA={lock['manifest_sha256']!r}
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def safe_unpack(archive,destination):
    destination.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        for i in z.infolist():
            p=Path(i.filename)
            assert not p.is_absolute() and '..' not in p.parts and chr(92) not in i.filename
            assert (destination/p).resolve().is_relative_to(destination.resolve())
            assert (i.external_attr>>16)&0o170000!=0o120000
        z.extractall(destination)
matches=list(Path('/kaggle/input').rglob('phase03_pgsr_source.bundle'))
assert len(matches)==1, matches
assert digest(matches[0])==SOURCE_SHA
safe_unpack(matches[0],WORK/'source')
SOURCE=WORK/'source/capsule'
assert digest(SOURCE/'manifest.json')==SOURCE_MANIFEST_SHA
manifest=json.loads((SOURCE/'manifest.json').read_text())
assert digest(SOURCE/'source.zip')==manifest['files']['source.zip']
safe_unpack(SOURCE/'source.zip',PROJECT)
os.chdir(PROJECT)
''')
    section("4. Locate the frozen prepared input","Reuse the existing private Phase02 input dataset. The PGSR adapter extracts and hashes only scene/; it never loads its B0 checkpoint or B0 source capsule.",'''
matches=list(Path('/kaggle/input').rglob('ignatius_phase02_input_v1.bundle'))
assert len(matches)==1, matches
PREPARED_ARCHIVE=matches[0]
print('Prepared transport:',PREPARED_ARCHIVE)
''')
    section("5. Build isolated PGSR CUDA environment","Pinned Python 3.10.20 / Torch 2.4.1 cu121 / CUDA toolkit 12.1.1. Build the PGSR plane rasterizer and its simple-knn. PyTorch3D provides its official pure-Torch CUDA transform; unused native PyTorch3D operations are not built.",'''
with Path('/kaggle/working/phase03_setup.log').open('w') as log:
    setup=subprocess.run(['bash','scripts/kaggle_pgsr/setup_linux.sh'],stdout=log,stderr=subprocess.STDOUT)
if setup.returncode:
    print(Path('/kaggle/working/phase03_setup.log').read_text()[-16000:])
    raise RuntimeError('PGSR CUDA setup failed; no CPU fallback')
PY=PROJECT/'.tools/pgsr_env/bin/python'
def command(*args):
    subprocess.run([str(PY),*map(str,args)],cwd=PROJECT,check=True)
''')
    section("6. Qualify the exact dataset and source","Prepared cameras, train-only point initialization and source hashes must match the sealed input. Optional resume accepts PGSR full state with unchanged optimization/source identity.",'''
args=['scripts/kaggle_pgsr/runtime.py','prepare','--source',SOURCE,'--archive',PREPARED_ARCHIVE]
if PREVIOUS is not None: args+=['--previous',PREVIOUS]
command(*args)
''')
    section("7. Real CUDA geometry and interruption/resume smoke","Synthetic plane/warp/gradient checks run first inside each training process. Then eight real Ignatius train views execute 80 iterations, restore all native Adam/ADC/exposure/RNG/sampler states, continue to 300 and verify PLY reload plus RGB/depth/normal outputs. This development test is not a baseline benchmark.",'''
command('scripts/kaggle_pgsr/runtime.py','smoke')
''')
    section("8. Research configuration","Same original 1920x1080 frames and frozen test split as B0. Official-code TNT settings and exposure enabled. ncc_scale=1 keeps grayscale at this full resolution. No silent OOM profile changes.",'''
command('-m','src.pgsr.runner','--config','configs/training/pgsr_research_quality.yaml','--dry-run')
''')
    section("9. Train PGSR and save recovery","Native optimization order and actual CUDA telemetry are logged. A session stops at a complete iteration before its declared time limit, publishes a full checkpoint and exports recovery before evaluation. Failures preserve the newest valid checkpoint.",'''
command('scripts/kaggle_pgsr/runtime.py','train','--output',OUTPUT)
print(json.loads((OUTPUT/'index.json').read_text()))
''')
    section("10. Held-out RGB and geometry export","At 30k only: raw canonical test RGB without exposure fitting, PSNR/SSIM/LPIPS on all 33 test views, separate native PLY reload verification, and positive axial-Z / camera-frame unit normal / valid-mask NPZ for every train view. These are Phase03 artifacts, not a mesh benchmark. Partial runs require another session first.",'''
command('scripts/kaggle_pgsr/runtime.py','evaluate','--output',OUTPUT)
''')
    section("11. Verify the saved handoff","Recovery and evaluation are sealed separately so a failed evaluator cannot invalidate the full training checkpoint. Only the latest research checkpoint is exported to avoid duplicating multi-gigabyte state.",'''
sys.path.insert(0,str(PROJECT/'scripts/kaggle_pgsr'))
from handoff import verify
recovery=verify(OUTPUT/'recovery')
assert recovery['method']=='B2_PGSR'
if (OUTPUT/'evaluation').exists():
    final=verify(OUTPUT/'evaluation')
    assert final['recovery_manifest_sha256']==digest(OUTPUT/'recovery/manifest.json')
print(json.loads((OUTPUT/'index.json').read_text()))
print('Published PGSR iteration:',recovery['iteration'])
''')
    section("12. Download or resume","Download phase03_results/ for local import and Blender review. If status is RECOVERY_SAVED below 30k, attach this notebook output as a Kernel Source in a new session and set PREVIOUS to the corresponding recovery/ path. Keep the same source capsule. Never select a B0 checkpoint.",'''
print('Output files:',sum(1 for p in OUTPUT.rglob('*') if p.is_file()))
print('Output GiB:',round(sum(p.stat().st_size for p in OUTPUT.rglob('*') if p.is_file())/1024**3,3))
print(json.dumps(json.loads((OUTPUT/'index.json').read_text()),indent=2))
''')
    notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
              "language_info":{"name":"python","version":"3.10"}},"nbformat":4,"nbformat_minor":5}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(notebook,indent=1)+"\n",encoding="utf-8")
    for cell in cells:
        if cell["cell_type"]=="code": compile("".join(cell["source"]),str(output),"exec")
    return {"sections":12,"code_cells_compiled":12,"notebook":str(output)}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--transport",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    print(build(args.transport,args.output))
