"""Independent CUDA finite-difference and invalid-domain checks for pinned PGSR."""
from pathlib import Path
import argparse
import math
import sys
import tempfile
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def check(output):
    from src.pgsr.native import initialize
    environment=initialize()
    import numpy as np
    import torch
    from PIL import Image
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel
    from scene.app_model import AppModel
    from gaussian_renderer import render
    from utils.graphics_utils import BasicPointCloud
    from src.pgsr.geometry import objectives,valid_geometry
    from src.pgsr.config import DEFAULTS
    from src.common.io import write_json
    output.mkdir(parents=True,exist_ok=False)
    Image.new('RGB',(48,48),(127,127,127)).save(output/'camera.png')
    xy=np.array([(x,y) for x in np.linspace(-.7,.7,6) for y in np.linspace(-.7,.7,6)])
    # Distinct depths avoid the renderer's discrete sort changing under +/- h.
    xyz=np.column_stack((xy,2.+.15*xy[:,0]+.07*xy[:,1]))
    model=GaussianModel(3)
    model.create_from_pcd(BasicPointCloud(xyz,np.full_like(xyz,.7),np.zeros_like(xyz)),1.)
    opt=SimpleNamespace(**DEFAULTS['optimizer'],iterations=10)
    opt.single_view_weight_from_iter=0
    opt.multi_view_weight_from_iter=0
    opt.multi_view_sample_num=128
    model.training_setup(opt)
    with torch.no_grad():
        model._scaling[:]=torch.tensor([math.log(.19),math.log(.23),math.log(.003)],device='cuda')
        model._opacity[:]=.5
        # Slant the Gaussian planes about Y with a unique minimum scale axis.
        model._rotation[:]=torch.tensor([math.cos(.1),0,math.sin(.1),0],device='cuda')
    camera=Camera(1,np.eye(3),np.zeros(3),math.pi/2,math.pi/2,48,48,str(output/'camera.png'),'unit',0,preload_img=False,data_device='cpu')
    pipe=SimpleNamespace(compute_cov3D_python=False,convert_SHs_python=False,debug=False)
    bg=torch.zeros(3,device='cuda')
    def scalar():
        p=render(camera,model,pipe,bg)
        return p['render'].mean()+.01*p['plane_depth'][0,24,24]+.01*p['rendered_normal'][0,24,24]
    value=scalar()
    value.backward()
    results={}
    for name,index,epsilon in [('xyz',(20,2),.0003),('scaling',(20,0),.002),('rotation',(20,2),.002),('opacity',(20,0),.002)]:
        param=getattr(model,'_'+name)
        analytic=float(param.grad[index])
        with torch.no_grad():
            original=float(param[index])
            param[index]=original+epsilon
            plus=float(scalar())
            param[index]=original-epsilon
            minus=float(scalar())
            param[index]=original
        numerical=(plus-minus)/(2*epsilon)
        error=abs(analytic-numerical)
        tolerance=max(2e-5,.08*max(abs(analytic),abs(numerical)))
        if error>tolerance or abs(analytic)<1e-7:
            raise AssertionError((name,analytic,numerical,error,tolerance))
        results[name]={'autograd':analytic,'finite_difference':numerical,'absolute_error':error,'tolerance':tolerance}
    model.optimizer.zero_grad(set_to_none=True)
    package=render(camera,model,pipe,bg)
    valid=valid_geometry(package,camera)
    assert valid.any()
    # Constant NCC patches must be rejected, even where geometric roundtrips pass.
    app=AppModel(1)
    gray=torch.ones(1,48,48,device='cuda')*.5
    target=package['render'].detach().clamp(0,1)
    total,terms,_,counts=objectives(model,app,camera,camera,package,package,target,gray,gray,opt,DEFAULTS,2)
    assert counts['multi_valid_pixels']>0 and counts['ncc_valid_patches']==0
    assert torch.isfinite(total) and terms['multi_ncc']==0
    # No geometric support yields connected finite zeros, not apparent convergence.
    empty={**package,'plane_depth':package['plane_depth']*0,'rendered_normal':package['rendered_normal']*0}
    total,terms,_,empty_counts=objectives(model,app,camera,camera,empty,empty,target,gray,gray,opt,DEFAULTS,2)
    assert empty_counts['geometry_valid_pixels']==empty_counts['multi_valid_pixels']==0
    assert torch.isfinite(total) and terms['single']==terms['multi_geo']==terms['multi_ncc']==0
    total.backward()
    # Exposure is gated off if canonical SSIM discrepancy is above native threshold.
    gated={**package,'render':torch.zeros_like(package['render']),'app_image':torch.ones_like(package['render'])}
    _,_,_,gate=objectives(model,app,camera,None,gated,None,torch.ones_like(target),gray,None,opt,DEFAULTS,0)
    assert not gate['exposure_gate_used']
    report={'status':'SUCCESS','scope':'SYNTHETIC_CUDA_NUMERICAL_CHECKS','environment':environment,
            'finite_difference':results,'constant_ncc_rejected':True,'empty_geometry_connected_zero':True,
            'exposure_high_structural_error_gate':True}
    write_json(output/'geometry_checks.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    print(check(parser.parse_args().output))
