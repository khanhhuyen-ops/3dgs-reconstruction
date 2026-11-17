"""PGSR safety contracts and opt-in actual CUDA checkpoint restore checks."""
from pathlib import Path
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.pgsr.config import load


@pytest.mark.parametrize('overrides',[
    {'device':'cpu'}, {'allow_cpu_fallback':True},
    {'output_root':'outputs/B0_3DGS'}, {'output_root':'../B2_PGSR'},
    {'optimizer':{'multi_view_weight_from_iter':1,'single_view_weight_from_iter':2}},
])
def test_incompatible_training_policy_rejected(tmp_path,overrides):
    config=tmp_path/'bad.yaml'
    config.write_text(yaml.safe_dump(overrides))
    with pytest.raises(ValueError): load(config)


def test_pgsr_bundle_rejects_b0_and_path_traversal(tmp_path):
    sys.path.insert(0,str(ROOT/'scripts/kaggle_pgsr'))
    from handoff import verify,write_json
    write_json(tmp_path/'manifest.json',{'schema':'phase02-kaggle-output-v1','files':{}})
    with pytest.raises(ValueError,match='PGSR'): verify(tmp_path)
    write_json(tmp_path/'manifest.json',{'schema':'phase03-pgsr-output-v1','files':{'../outside':'0'*64}})
    with pytest.raises(ValueError,match='Unsafe'): verify(tmp_path)


@pytest.mark.skipif(os.environ.get('PGSR_RUN_CUDA_TESTS')!='1',reason='Requires isolated PGSR CUDA env and completed real-data smoke')
def test_actual_smoke_full_state_and_rng_restore(tmp_path):
    from src.pgsr.native import initialize
    environment=initialize()
    import torch
    from scene.gaussian_model import GaussianModel
    from scene.app_model import AppModel
    from src.pgsr import checkpoint
    from src.common.io import read_json
    from src.pgsr.validation import validate_model
    run=ROOT/os.environ.get('PGSR_SMOKE_RUN','outputs/tanks_and_temples/Ignatius/research/B2_PGSR/local-smoke-v1')
    config=read_json(run/'resolved_config.json')
    identity=read_json(run/'identity.json')
    pointer=read_json(run/'checkpoints/latest.json')
    inspection=checkpoint.inspect(run/'checkpoints'/pointer['path'],identity)
    assert inspection['iteration']==300 and inspection['optimizer_steps']==299
    model=GaussianModel(config['sh_degree'])
    app=AppModel(config['train_view_limit'])
    options=SimpleNamespace(**config['optimizer'],iterations=config['iterations'])
    train_ids=read_json(ROOT/config['dataset']/'splits.json')['image_ids']['train'][:config['train_view_limit']]
    payload=checkpoint.restore(run,model,app,options,identity,environment,train_ids)
    python_rng=random.Random();python_rng.setstate(payload['rng']['python'])
    numpy_rng=np.random.RandomState();numpy_rng.set_state(payload['rng']['numpy'])
    cpu_rng=torch.Generator();cpu_rng.set_state(payload['rng']['torch_cpu'].cpu())
    cuda_rng=torch.Generator(device='cuda');cuda_rng.set_state(payload['rng']['torch_cuda'][0].cpu())
    assert random.random()==python_rng.random()
    assert np.random.random()==numpy_rng.random_sample()
    assert torch.equal(torch.rand(10),torch.rand(10,generator=cpu_rng))
    assert torch.equal(torch.rand(10,device='cuda'),torch.rand(10,device='cuda',generator=cuda_rng))
    capture=model.capture()
    for i in range(1,14): assert torch.equal(capture[i],payload['native_capture'][i])
    assert torch.equal(app.appear_ab,payload['app_state']['appear_ab'])
    assert app.optimizer.state_dict()['state']
    assert validate_model(model)['gaussians']==inspection['gaussians']
    with pytest.raises(ValueError,match='identity'):
        checkpoint.inspect(run/'checkpoints'/pointer['path'],{'method':'B0'})
    wrong=tmp_path/'b0.pth'
    torch.save({'schema':'b0-checkpoint-v1'},wrong)
    with pytest.raises(ValueError,match='B0/PLY'): checkpoint.inspect(wrong)
