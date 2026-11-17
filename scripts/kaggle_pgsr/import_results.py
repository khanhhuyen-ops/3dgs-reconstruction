"""Validate PGSR full state, sealed final metrics/geometry, then import without overwrite."""
import argparse
import math
from pathlib import Path
import shutil
import sys

from handoff import ROOT,verify,read_json,write_json,sha256,inside


def import_results(folder):
    from src.pgsr.native import initialize,code_identity
    initialize()  # Own CUDA env/DLLs; serialization itself is inspected on CPU.
    from src.pgsr.checkpoint import inspect
    from src.common.io import fingerprint
    import numpy as np
    from plyfile import PlyData
    recovery=folder/'recovery'
    manifest=verify(recovery)
    if manifest['schema']!='phase03-pgsr-output-v1' or manifest['method']!='B2_PGSR':
        raise ValueError('Expected PGSR recovery')
    run=recovery/'run'
    config=read_json(run/'resolved_config.json')
    identity=read_json(run/'identity.json')
    dataset=inside(ROOT,config['dataset'])
    split=read_json(dataset/'splits.json')
    if identity['code']!=code_identity() or identity['config']!=fingerprint(config):
        raise ValueError('Imported PGSR source/optimization config mismatch')
    if identity['metadata']!=sha256(dataset/'metadata.json') or identity['split']!=split['hash']:
        raise ValueError('Imported PGSR dataset/split mismatch')
    pointer=read_json(run/'checkpoints/latest.json')
    checkpoint=inside(run/'checkpoints',pointer['path'])
    if sha256(checkpoint)!=pointer['sha256']:
        raise ValueError('Imported PGSR checkpoint checksum mismatch')
    validation=inspect(checkpoint,identity)
    import torch
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    cap=payload['native_capture'];n=validation['gaussians']
    if not 0<=cap[0]<=config['sh_degree'] or not math.isfinite(cap[15]) or cap[15]<=0:
        raise ValueError('Invalid PGSR SH/spatial extent')
    for index in range(8,14):
        expected=(n,) if index in (8,9) else (n,1)
        if tuple(cap[index].shape)!=expected:
            raise ValueError('PGSR ADC/radii accumulator shape mismatch')
    expected_groups={'xyz','knn_f','f_dc','f_rest','opacity','scaling','rotation'}
    if {g['name'] for g in cap[14]['param_groups']}!=expected_groups:
        raise ValueError('PGSR optimizer groups differ')
    train_ids=split['image_ids']['train']
    if validation['exposure_rows']!=len(train_ids) or not set(payload['sampler_remaining_ids'])<=set(train_ids):
        raise ValueError('PGSR exposure/sampler references another training set')
    del payload,cap
    if validation!=manifest['checkpoint_validation'] or validation['iteration']!=manifest['iteration']:
        raise ValueError('PGSR full-state manifest mismatch')
    evaluated=False
    evaluation_manifest=None
    if (folder/'evaluation').exists():
        evaluation_manifest=verify(folder/'evaluation')
        if evaluation_manifest['schema']!='phase03-pgsr-evaluation-v1' or not evaluation_manifest['complete']:
            raise ValueError('Expected complete PGSR evaluation bundle')
        if evaluation_manifest['recovery_manifest_sha256']!=sha256(recovery/'manifest.json') or evaluation_manifest['source_manifest_sha256']!=manifest['source_manifest_sha256']:
            raise ValueError('PGSR final evaluation belongs to another recovery')
        result=read_json(run/'training_result.json')
        if result['status']!='SUCCESS' or result['iterations']!=config['iterations'] or validation['iteration']!=30000:
            raise ValueError('Partial training mislabeled final')
        evaluation=folder/'evaluation/evaluation'
        metrics=read_json(evaluation/'rendering_metrics.json')
        geometry=read_json(evaluation/'geometry_contract.json')
        reload=read_json(evaluation/'reload_validation.json')
        if metrics['status']!='SUCCESS' or [v['image_id'] for v in metrics['views']]!=split['image_ids']['test'] or metrics['test_exposure_fit']:
            raise ValueError('Invalid frozen held-out PGSR evaluation')
        if any(not math.isfinite(v[k]) for v in metrics['views'] for k in ('psnr','ssim','lpips')):
            raise ValueError('Nonfinite PGSR metrics')
        if metrics['dataset_identity']!={k:identity[k] for k in ('metadata','split')}:
            raise ValueError('PGSR metrics dataset differs')
        if reload['status']!='SUCCESS' or max(reload['max_abs_errors'].values())>1e-5:
            raise ValueError('PGSR PLY reload failed')
        if geometry['status']!='SUCCESS' or [v['image_id'] for v in geometry['views']]!=split['image_ids']['train']:
            raise ValueError('Missing train-view geometry')
        metadata={row['image_id']:row for row in read_json(dataset/'metadata.json')['images']}
        for row in geometry['views']:
            path=inside(evaluation,row['path'])
            if sha256(path)!=row['sha256']:
                raise ValueError('PGSR geometry hash mismatch')
            with np.load(path,allow_pickle=False) as arrays:
                depth,normal,valid=arrays['depth'],arrays['normal'],arrays['valid']
                source=metadata[row['image_id']]
                shape=(round(source['height']/config['resolution']),round(source['width']/config['resolution']))
                if depth.shape!=shape or normal.shape!=(*shape,3) or valid.shape!=shape or valid.dtype!=np.bool_:
                    raise ValueError('PGSR geometry layout mismatch')
                if not valid.any() or not np.isfinite(depth).all() or not np.isfinite(normal).all() or not (depth[valid]>0).all():
                    raise ValueError('Invalid PGSR depth/normal data')
                if not np.allclose(np.linalg.norm(normal[valid],axis=1),1,atol=1e-5) or np.any(depth[~valid]) or np.any(normal[~valid]):
                    raise ValueError('PGSR unit-normal/invalid mask mismatch')
                if not np.allclose(arrays['Tcw'],source['Tcw'],atol=1e-6):
                    raise ValueError('PGSR geometry camera pose changed')
        ply=run/'point_cloud'/f"iteration_{validation['iteration']:06d}"/'point_cloud.ply'
        if sha256(ply)!=result['export']['sha256']:
            raise ValueError('PGSR export checksum mismatch')
        parsed=PlyData.read(ply)
        names=set(parsed['vertex'].data.dtype.names)
        required={'x','y','z','opacity'}|{f'scale_{i}' for i in range(3)}|{f'rot_{i}' for i in range(4)}|{f'f_dc_{i}' for i in range(3)}|{f'f_rest_{i}' for i in range(3*((config['sh_degree']+1)**2-1))}
        if not required<=names or len(parsed['vertex'])!=validation['gaussians'] or any(not np.isfinite(parsed['vertex'].data[n]).all() for n in names):
            raise ValueError('Invalid PGSR Gaussian PLY')
        evaluated=True
    tag=fingerprint({'recovery':sha256(recovery/'manifest.json'),'evaluation':evaluation_manifest})[:10]
    output_root=inside(ROOT,config['output_root'])
    if not output_root.is_relative_to(ROOT/'outputs') or 'B2_PGSR' not in output_root.parts:
        raise ValueError('Import must stay in B2_PGSR output tree')
    target=inside(output_root,manifest['run_name']+'-import-'+tag)
    receipt={'status':'IMPORTED_FINAL' if evaluated else 'IMPORTED_PARTIAL','bundle_id':tag,
             'recovery_manifest_sha256':sha256(recovery/'manifest.json'),'checkpoint':validation,
             'evaluation_verified':evaluated,'source_manifest_sha256':manifest['source_manifest_sha256']}
    if target.exists():
        if read_json(target/'kaggle_import.json')!=receipt:
            raise FileExistsError(target)
        for name,digest in manifest['files'].items():
            if name.startswith('run/') and sha256(inside(target,name[4:]))!=digest:
                raise ValueError('Previously imported PGSR file changed')
        if evaluation_manifest:
            for name,digest in evaluation_manifest['files'].items():
                if sha256(inside(target,name))!=digest:
                    raise ValueError('Previously imported PGSR evaluation changed')
        return target
    temporary=target.with_name('.'+target.name+'.importing')
    shutil.copytree(run,temporary)
    if evaluated:
        shutil.copytree(folder/'evaluation/evaluation',temporary/'evaluation')
    write_json(temporary/'kaggle_import.json',receipt)
    temporary.rename(target)
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,required=True)
    print(import_results(parser.parse_args().results.resolve()))
