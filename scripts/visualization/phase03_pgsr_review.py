"""Reuse the existing point-cloud importer for a separate PGSR Blender review scene."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from phase02_pointcloud_review import prepare,build_blender


def prepare_review(run: Path,output: Path):
    from src.common.io import read_json,write_json
    result=read_json(run/'training_result.json')
    if result['status']!='SUCCESS':
        raise ValueError('Review requires a completed declared PGSR profile')
    validation=read_json(run/'evaluation/reload_validation.json')
    if validation['status']!='SUCCESS' or max(validation['max_abs_errors'].values())>1e-5:
        raise ValueError('PGSR native PLY reload has not passed')
    ply=run/'point_cloud'/f"iteration_{result['iterations']:06d}"/'point_cloud.ply'
    report=prepare(ply,output,'PGSR_B2',result['export']['sha256'])
    receipt={'method':'B2_PGSR','run':run.relative_to(ROOT).as_posix(),
             'report':report.relative_to(ROOT).as_posix(),'iterations':result['iterations'],
             'development_only':result['iterations']!=30000,
             'dataset':read_json(run/'resolved_config.json')['dataset']}
    write_json(output/'handoff.json',receipt)
    return receipt


def build_review(handoff: str,destination: str):
    """Run inside the existing Blender/addon, preserving every existing object."""
    import bpy
    from src.common.io import read_json,write_json
    receipt=read_json(ROOT/handoff)
    run=ROOT/receipt['run']
    result=build_blender([receipt['report']],receipt['dataset'],destination)
    scene=bpy.data.scenes[result['scene']]
    scene.name='Phase03_PGSR_Review'
    scene.world.name='Phase03_PGSR_World'
    scene.camera.name='Phase03_PGSR_COLMAP_TestCamera'
    scene['method']='B2_PGSR'
    scene['source_run']=receipt['run']
    scene['development_only']=receipt['development_only']
    # Pack the actual CUDA RGB render and plane-depth/normal preview images.
    images=[run/'evaluation/test_rgb/000001.png']
    images+=sorted((run/'evaluation/geometry_preview').glob('*.png'))[:6]
    packed=[]
    for path in images:
        image=bpy.data.images.load(str(path),check_existing=False)
        image.name='PGSR_'+path.name
        image.pack()
        packed.append(image.name)
    for area in bpy.context.screen.areas:
        if area.type=='IMAGE_EDITOR' and packed:
            area.spaces.active.image=bpy.data.images[packed[0]]
    result.update(scene=scene.name,method='B2_PGSR',camera=scene.camera.name,
                  packed_native_images=packed,development_only=receipt['development_only'],
                  source_run=receipt['run'],note='Existing importer reused; B0 objects/coordinates preserved; viewport sample is not an official metric render')
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/destination),check_existing=False)
    write_json((ROOT/destination).with_suffix('.json'),result)
    print(json.dumps(result))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    print(prepare_review(args.run.resolve(),args.output.resolve()))
