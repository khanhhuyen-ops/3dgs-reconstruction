"""Build a standalone inference-only notebook; validation never executes its cells."""
from pathlib import Path
import base64
import hashlib
import io
import json
import zipfile

ROOT=Path(__file__).resolve().parents[2]

WORKER=r'''
import os,sys,json,gc,hashlib,importlib.util,_codecs
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import open3d as o3d
from torch.torch_version import TorchVersion
from src.common.io import sha256,read_json,write_json
from src.pgsr.native import initialize
from src.pgsr.data import SceneAdapter
settings=read_json(Path(sys.argv[1]));output=Path(settings['output'])
env=initialize()
if not o3d.core.cuda.is_available():raise RuntimeError('BLOCKED_BACKEND: Open3D CUDA unavailable; no CPU fusion fallback')
device=o3d.core.Device('CUDA:0')
probe=o3d.core.Tensor([1.],device=device)+1
assert float(probe.cpu().numpy()[0])==2.
input_root=Path('/kaggle/input');candidates=[]
for config_path in input_root.rglob('resolved_config.json'):
    cfg=read_json(config_path)
    if cfg.get('method')!='B2_PGSR':continue
    run=config_path.parent.parent if config_path.parent.name=='configs' else config_path.parent
    pointer=run/'checkpoints/latest.json'
    if pointer.exists():
        p=read_json(pointer);cp=run/'checkpoints'/p['path']
        if cp.parent.resolve()!=(run/'checkpoints').resolve():raise ValueError('Unsafe checkpoint pointer')
        if cp.is_file() and p['iteration']>300:candidates.append((p['iteration'],cp,cfg,p,run))
if settings.get('checkpoint'):
    candidates=[x for x in candidates if str(x[1])==settings['checkpoint']]
if not candidates:raise FileNotFoundError('Attach the extracted original PGSR notebook output (recovery/run/checkpoints and config), not B0 or a ZIP-only upload')
candidates.sort(key=lambda x:(-x[0],str(x[1])))
iteration,checkpoint,cfg,pointer,run=candidates[0]
if sha256(checkpoint)!=pointer['sha256']:raise ValueError('Checkpoint SHA256 mismatch')
if settings.get('expected_checkpoint_sha256') and pointer['sha256']!=settings['expected_checkpoint_sha256']:
    raise ValueError('Mounted model differs from the configured verified recovery')
if any(x[0]==iteration and x[3]['sha256']!=pointer['sha256'] for x in candidates):
    raise ValueError('Ambiguous equally recent PGSR models: set CHECKPOINT explicitly')
safe=[np.core.multiarray._reconstruct,np.core.multiarray.scalar,np.ndarray,np.dtype,TorchVersion,_codecs.encode]
safe += [type(np.dtype(t)) for t in (np.uint32,np.int32,np.int64,np.float32,np.float64)]
torch.serialization.add_safe_globals(safe)
try:payload=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
finally:torch.serialization.clear_safe_globals()
if payload.get('schema')!='pgsr-full-checkpoint-v1' or payload['iteration']!=iteration:raise ValueError('Not a compatible full PGSR checkpoint')
from src.common.io import fingerprint
if payload['identity']['config']!=fingerprint(cfg) or payload['identity']['native_revision']!=env['native_revision']:
    raise ValueError('Checkpoint configuration/native revision mismatch')
if iteration<cfg['iterations'] and not settings['allow_partial']:raise ValueError('Partial model; ALLOW_PARTIAL must be explicit')
dataset=Path.cwd()/cfg['dataset']
if not dataset.exists():
    matches=[]
    for m in input_root.rglob('metadata.json'):
        if sha256(m)==payload['identity']['metadata'] and (m.parent/'integrity.json').exists():matches.append(m.parent)
    if matches:
        dataset.parent.mkdir(parents=True,exist_ok=True);dataset.symlink_to(matches[0],target_is_directory=True)
    else:
        sys.path.insert(0,str(Path.cwd()/'scripts/kaggle_pgsr'))
        from handoff import extract_scene
        archives=list(input_root.rglob('ignatius_phase02_input_v1.bundle'))
        if len(archives)!=1:raise FileNotFoundError('Attach existing Ignatius Phase02 prepared input dataset or the exact extracted prepared scene')
        extract_scene(archives[0],dataset)
data=SceneAdapter(cfg,output/'qualification')
if data.identity!={k:payload['identity'][k] for k in ('metadata','split')}:raise ValueError('Frozen data/camera split differs')
from scene.gaussian_model import GaussianModel
model=GaussianModel(cfg['sh_degree']);cap=payload['native_capture']
if len(cap)!=16:raise ValueError('Native PGSR capture mismatch')
for name,index in (('_xyz',1),('_features_dc',3),('_features_rest',4),('_scaling',5),('_rotation',6),('_opacity',7)):
    value=cap[index]
    if not torch.isfinite(value).all():raise ValueError('Nonfinite model parameters')
    setattr(model,name,torch.nn.Parameter(value.to('cuda:0'),requires_grad=False))
model.active_sh_degree=cap[0];model.use_app=False
del cap,payload;gc.collect();torch.cuda.empty_cache()
# Import the pinned official renderer/extraction loop, bypassing its destructive Scene reader.
spec=importlib.util.spec_from_file_location('official_pgsr_render',Path('external/PGSR/render.py'))
official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
original_render=official.render
maps=output/'maps';maps.mkdir()
def export_render(camera,*args,**kwargs):
    package=original_render(camera,*args,**kwargs)
    z=package['plane_depth'].squeeze();normal=package['rendered_normal']
    norm=normal.norm(dim=0)
    valid=torch.isfinite(z)&(z>0)&(z<settings['max_depth'])&torch.isfinite(normal).all(0)&(norm>1e-6)
    # Same native 80-degree grazing-angle filter, plus explicit undefined-value masks.
    ray=torch.nn.functional.normalize(camera.get_rays(),dim=-1)
    dn=torch.nn.functional.normalize(package['depth_normal'].permute(1,2,0),dim=-1)
    valid &= (ray*dn).sum(-1).abs()>=np.cos(np.deg2rad(80.))
    package['plane_depth']=torch.where(valid[None],package['plane_depth'],0)
    unit=torch.where(valid[None],normal/norm.clamp_min(1e-8)[None],0)
    np.savez_compressed(maps/f'{camera.image_id:06d}.npz',depth=package['plane_depth'].squeeze().cpu().numpy(),
        normal=unit.permute(1,2,0).cpu().numpy(),valid=valid.cpu().numpy(),
        K=camera.get_k().cpu().numpy(),Tcw=camera.world_view_transform.T.cpu().numpy())
    return package
official.render=export_render
class CudaVolume:
    def __init__(self):
        self.grid=o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf','weight','color'),
            attr_dtypes=(o3d.core.float32,)*3,attr_channels=(1,1,3),voxel_size=settings['voxel_size'],
            block_resolution=16,block_count=settings['block_count'],device=device)
        assert str(self.grid.attribute('tsdf').device)=='CUDA:0'
        self.frames=0
    def integrate(self,rgbd,intrinsic,pose):
        # Native RGBD decoding is I/O; TSDF allocation/integration/extraction stay on CUDA.
        depth=o3d.t.geometry.Image(o3d.core.Tensor(np.asarray(rgbd.depth).copy(),device=device))
        color=o3d.t.geometry.Image(o3d.core.Tensor(np.asarray(rgbd.color).copy(),device=device))
        k=o3d.core.Tensor(intrinsic.intrinsic_matrix,o3d.core.float64)
        t=o3d.core.Tensor(pose,o3d.core.float64)
        blocks=self.grid.compute_unique_block_coordinates(depth,k,t,1.,settings['max_depth'],4.)
        self.grid.integrate(blocks,depth,color,k,k,t,1.,settings['max_depth'],4.)
        self.frames+=1
        if self.grid.hashmap().size()>settings['block_count']:raise RuntimeError('Declared voxel block budget exceeded')
    def extract(self):
        mesh=self.grid.extract_triangle_mesh(weight_threshold=1.)
        if str(mesh.device)!='CUDA:0':raise RuntimeError('Extraction did not return a CUDA mesh')
        mesh.compute_vertex_normals()
        return mesh.cpu().to_legacy()
volume=CudaVolume()
pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
bg=torch.full((3,),float(cfg['white_background']),device='cuda:0')
with torch.no_grad():
    official.render_set(str(output),'train',iteration,data.train,None,model,pipe,bg,
        app_model=None,max_depth=settings['max_depth'],volume=volume,use_depth_filter=True)
    mesh=volume.extract()
vertices=np.asarray(mesh.vertices);triangles=np.asarray(mesh.triangles)
if not len(vertices) or not len(triangles) or not np.isfinite(vertices).all():raise ValueError('Empty/nonfinite reconstructed mesh')
if triangles.min()<0 or triangles.max()>=len(vertices):raise ValueError('Invalid triangle indices')
target=output/'mesh.ply'
if not o3d.io.write_triangle_mesh(str(target),mesh):raise RuntimeError('Mesh export failed')
reopened=o3d.io.read_triangle_mesh(str(target))
if len(reopened.triangles)!=len(triangles):raise ValueError('Export reopen validation failed')
report={'status':'SUCCESS','method':'PGSR official render_set + CUDA tensor TSDF adapter',
    'native_revision':env['native_revision'],'checkpoint_sha256':pointer['sha256'],'iteration':iteration,
    'training_complete':iteration==cfg['iterations'],'training_executed':False,'environment':env,
    'vertices':len(vertices),'triangles':len(triangles),'integrated_train_views':volume.frames,
    'voxel_size':settings['voxel_size'],'sdf_trunc':4*settings['voxel_size'],
    'depth_max':settings['max_depth'],'units':'Original arbitrary monocular SfM units; no GT alignment',
    'maps':'positive axial Z, camera-frame unit normals, invalid zeros + boolean mask; K/Tcw included',
    'deviations':'CUDA tensor volume replaces upstream legacy CPU volume; finite/support masks added; no largest-component cleaning; selected resource voxel preset',
    'chamfer_distance':None,'f_score':None,'geometry_evaluation':'NOT AVAILABLE: no qualified GT protocol',
    'mesh_sha256':sha256(target),'peak_torch_reserved_bytes':torch.cuda.max_memory_reserved(),
    'note':'Torch memory excludes Open3D allocations; no held-out RGB fitted or used for fusion'}
write_json(output/'mesh_metrics.json',report)
print(json.dumps(report,indent=2))
'''


def build() -> Path:
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in ('src','scripts/kaggle','scripts/kaggle_pgsr'):
            for p in sorted((ROOT/folder).rglob('*')):
                if p.is_file() and p.suffix in ('.py','.sh') and '__pycache__' not in p.parts:
                    z.writestr(p.relative_to(ROOT).as_posix(),p.read_text(encoding='utf-8').replace('\r\n','\n'))
    blob=buffer.getvalue();encoded=base64.b64encode(blob).decode();checksum=hashlib.sha256(blob).hexdigest()
    cells=[]
    def cell(kind,text):
        c={'cell_type':kind,'id':f'mesh-{len(cells):02d}','metadata':{},'source':text.strip().splitlines(True)}
        if kind=='code':c.update(execution_count=None,outputs=[])
        cells.append(c)
    cell('markdown','''# PGSR mesh reconstruction — Kaggle Run All
Enable **GPU T4** and **Internet**. Attach the existing PGSR notebook output (extracted `phase03_results/recovery/run`) and the private `ignatius-phase02-3dgs-resume-input-v1` dataset, or the exact extracted prepared scene. No source dataset or manual clone is needed: project helpers are embedded and upstream native code is pinned.

No training is called. Auto-selection chooses the latest research PGSR full checkpoint; ambiguous equal-iteration models fail safely. The known recovery is **partial at21298/30000** and is accepted explicitly below. Output is a mesh of that saved state, not a claim of completed training or benchmark quality.

Upstream PGSR uses CPU legacy Open3D TSDF. To honor strict GPU policy this notebook reuses official `render.py::render_set` and replaces only its volume with CUDA tensor TSDF integration/extraction. This is a declared backend adaptation, not bitwise upstream reproduction. CPU handles I/O and mesh serialization. CUDA-unavailable builds fail without fallback. Defaults are resource presets in arbitrary SfM units; refine only as a declared extraction variant.
''')
    cell('code',r'''
import os,sys,json,base64,hashlib,io,zipfile,subprocess,shutil
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='0'
PROJECT=Path('/tmp/pgsr-mesh-project')
OUTPUT=Path('/kaggle/working/pgsr_mesh_results')
CHECKPOINT=None  # Optional absolute mounted path when equally recent models differ.
ALLOW_PARTIAL=True  # Explicitly accept the known 21298-step recovery without retraining.
VOXEL_SIZE=0.01  # Scene units; upstream CLI default is0.002, which needs more memory.
MAX_DEPTH=5.0  # Upstream default, arbitrary SfM units; not meters without alignment.
BLOCK_COUNT=30000
EXPECTED_CHECKPOINT_SHA256=None
configuration=list(Path('/kaggle/input').rglob('pgsr_mesh_settings.json'))
assert len(configuration)<=1, 'Multiple extraction configuration datasets attached'
if configuration:
    selected=json.loads(configuration[0].read_text())
    VOXEL_SIZE=selected['voxel_size'];MAX_DEPTH=selected['max_depth'];BLOCK_COUNT=selected['block_count']
    ALLOW_PARTIAL=selected['allow_partial']
    EXPECTED_CHECKPOINT_SHA256=selected.get('checkpoint_sha256')
assert VOXEL_SIZE>0 and 0<MAX_DEPTH<65.535 and BLOCK_COUNT>0
assert not PROJECT.exists() and not OUTPUT.exists(), 'Use a fresh session; existing results are preserved.'
OUTPUT.mkdir(parents=True)
subprocess.run(['nvidia-smi'],check=True)
''')
    cell('markdown','## Restore embedded project helpers and set up the pinned CUDA environment')
    cell('code',f'''
blob=base64.b64decode({encoded!r})
assert hashlib.sha256(blob).hexdigest()=={checksum!r}
PROJECT.mkdir(parents=True)
with zipfile.ZipFile(io.BytesIO(blob)) as z:
    for item in z.infolist():
        p=Path(item.filename)
        assert not p.is_absolute() and '..' not in p.parts and chr(92) not in item.filename
        assert (item.external_attr>>16)&0o170000!=0o120000
    z.extractall(PROJECT)
os.chdir(PROJECT)
with (OUTPUT/'setup.log').open('w') as log:
    result=subprocess.run(['bash','scripts/kaggle_pgsr/setup_linux.sh'],stdout=log,stderr=subprocess.STDOUT)
if result.returncode:
    print((OUTPUT/'setup.log').read_text()[-12000:]);raise RuntimeError('Pinned PGSR CUDA build failed')
PY=PROJECT/'.tools/pgsr_env/bin/python'
with (OUTPUT/'open3d_setup.log').open('w') as log:
    subprocess.run([str(PY),'-m','pip','install','open3d==0.19.0'],stdout=log,stderr=subprocess.STDOUT,check=True)
''')
    cell('markdown','## Locate checkpoint/data, render train views, and extract the mesh\nFloat depth/normal NPZ files and masks accompany upstream JPEG previews. All components are preserved; no largest-component-only cleaning is applied. Optimizer state stays off the GPU and is not stepped.')
    cell('code',f"worker={WORKER!r}\n(PROJECT/'mesh_worker.py').write_text(worker,encoding='utf-8')")
    cell('code',r'''
settings={'output':str(OUTPUT),'checkpoint':CHECKPOINT,'allow_partial':ALLOW_PARTIAL,
          'voxel_size':VOXEL_SIZE,'max_depth':MAX_DEPTH,'block_count':BLOCK_COUNT,
          'expected_checkpoint_sha256':EXPECTED_CHECKPOINT_SHA256}
settings_path=OUTPUT/'extraction_config.json'
settings_path.write_text(json.dumps(settings,indent=2))
with (OUTPUT/'extraction.log').open('w') as log:
    result=subprocess.run([str(PY),str(PROJECT/'mesh_worker.py'),str(settings_path)],stdout=log,stderr=subprocess.STDOUT)
if result.returncode:
    (OUTPUT/'failure.json').write_text(json.dumps({'status':'FAILED','returncode':result.returncode,'training_executed':False}))
    print((OUTPUT/'extraction.log').read_text()[-16000:])
    raise RuntimeError('Extraction failed; logs and completed maps are preserved. No CPU fallback or retraining.')
print((OUTPUT/'mesh_metrics.json').read_text())
''')
    cell('markdown','## Preserve available historical evaluation and package essential results\nNo new rendering-quality or GT geometry benchmark is invented. Historical metrics remain labeled with their original provenance. The ZIP excludes checkpoints, environments and dataset copies.')
    cell('code',r'''
historical=OUTPUT/'historical_metrics';historical.mkdir()
records=[]
for p in Path('/kaggle/input').rglob('rendering_metrics*.json'):
    # Only copy PGSR records with explicit method provenance.
    try:d=json.loads(p.read_text())
    except (ValueError,OSError):continue
    if d.get('method')!='B2_PGSR':continue
    name=hashlib.sha256(str(p).encode()).hexdigest()[:10]+'_'+p.name
    shutil.copyfile(p,historical/name)
    records.append({'source':str(p),'copied_as':name,'scope':'Previously recorded; may be a different iteration/profile; not an evaluation of the newly extracted mesh'})
(historical/'provenance.json').write_text(json.dumps(records,indent=2))
manifest={}
for p in OUTPUT.rglob('*'):
    if p.is_file():
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        manifest[p.relative_to(OUTPUT).as_posix()]=h.hexdigest()
(OUTPUT/'manifest.json').write_text(json.dumps({'files':manifest,'training_executed':False},indent=2))
size=sum(p.stat().st_size for p in OUTPUT.rglob('*') if p.is_file())
if size>8*1024**3 or shutil.disk_usage(OUTPUT.parent).free<size*1.05:
    raise RuntimeError('ZIP safety budget exceeded; files remain available individually. Use a declared coarser extraction profile in a fresh session.')
archive=Path(shutil.make_archive('/kaggle/working/pgsr_mesh_results','zip',OUTPUT))
from IPython.display import FileLink,display
display(FileLink(str(archive)))
print('Download:',archive,'bytes:',archive.stat().st_size)
''')
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'accelerator':'GPU'},'nbformat':4,'nbformat_minor':5}
    destination=ROOT/'pgsr_mesh_kaggle.ipynb'
    destination.write_text(json.dumps(notebook,indent=1)+'\n',encoding='utf-8')
    return destination

if __name__=='__main__':print(build())
