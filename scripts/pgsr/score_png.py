"""Primary B0-comparable PGSR scores from saved rounded PNGs; no model/appearance fitting."""
import argparse
import ast
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def score(run):
    from src.pgsr.native import initialize
    environment=initialize()
    import numpy as np
    import torch
    from PIL import Image
    from scene.cameras import process_image
    from src.common.io import read_json,write_json,sha256
    from src.evaluation.rendering_metrics import Metrics
    config=read_json(run/'resolved_config.json')
    evaluation=run/'evaluation'
    output=evaluation/'rendering_metrics_png.json'
    if output.exists(): raise FileExistsError(output)
    original=read_json(evaluation/'rendering_metrics.json')
    dataset=ROOT/config['dataset']
    split=read_json(dataset/'splits.json')
    metadata={r['image_id']:r for r in read_json(dataset/'metadata.json')['images']}
    integrity=read_json(dataset/'integrity.json')
    if [v['image_id'] for v in original['views']]!=split['image_ids']['test']:
        raise ValueError('PGSR PNG list differs from frozen test split')
    # Prove metric functions match the selected B0 source, rather than relying on names.
    sources=[ROOT/'external/gaussian-splatting/utils/loss_utils.py',ROOT/'external/PGSR/utils/loss_utils.py']
    functions=[]
    for source in sources:
        tree=ast.parse(source.read_text(encoding='utf-8'))
        functions.append({n.name:ast.dump(n,include_attributes=False) for n in tree.body
                          if isinstance(n,ast.FunctionDef) and n.name in ('gaussian','create_window','_ssim','ssim')})
    if functions[0]!=functions[1] or len(functions[0])!=4:
        raise ValueError('SSIM definitions differ from B0')
    metrics=Metrics()
    if metrics.lpips is None: raise RuntimeError(metrics.lpips_error)
    rows=[]
    references=evaluation/'test_reference_png';references.mkdir(exist_ok=False)
    with torch.no_grad():
        for iid in split['image_ids']['test']:
            row=metadata[iid]
            source=dataset/row['path']
            if sha256(source)!=integrity[row['path']]: raise ValueError('Prepared test image changed')
            size=(round(row['width']/config['resolution']),round(row['height']/config['resolution']))
            target,_,_=process_image(str(source),size,config['ncc_scale'])
            target_bytes=np.rint(target.permute(1,2,0).numpy()*255).astype(np.uint8)
            target_path=references/f'{iid:06d}.png'
            Image.fromarray(target_bytes).save(target_path)
            prediction_path=evaluation/'test_rgb'/f'{iid:06d}.png'
            with Image.open(prediction_path) as prediction:
                pred=np.array(prediction.convert('RGB'),dtype=np.float32).transpose(2,0,1)/255.
            target=target_bytes.astype(np.float32).transpose(2,0,1)/255.
            values=metrics(torch.from_numpy(pred).to('cuda:0'),torch.from_numpy(target).to('cuda:0'))
            rows.append({'image_id':iid,**values,'prediction_sha256':sha256(prediction_path),'reference_sha256':sha256(target_path)})
    report={'status':'SUCCESS','method':'B2_PGSR','split':'test','count':len(rows),'views':rows,
            'mean':{k:float(np.mean([r[k] for r in rows])) for k in ('psnr','ssim','lpips')},
            'color_space':'sRGB [0,1] from rounded uint8 PNG; same B0 evaluation protocol',
            'test_exposure_fit':False,'resolution_divisor':config['resolution'],
            'ssim_definition_ast_identical_to_b0':True,'weight_hashes':metrics.weight_hashes,
            'dataset_identity':original['dataset_identity'],'ply_sha256':read_json(run/'training_result.json')['export']['sha256'],
            'floating_render_metrics':'rendering_metrics.json (diagnostic; PNG report is primary for B0 comparison)',
            'evaluator_source_sha256':sha256(Path(__file__)),'environment':environment}
    write_json(output,report)
    return report['mean']


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    print(score(p.parse_args().run.resolve()))
