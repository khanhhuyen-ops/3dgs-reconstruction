"""Versioned official-code PGSR profiles, with explicit project overrides."""
from copy import deepcopy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    "method": "B2_PGSR", "variant": "official_tnt_shared_fullres_guarded_v1",
    "dataset": "datasets/tanks_and_temples/Ignatius/prepared/legacy_v1/ignatius-legacy-v1-6a3308c22df11122",
    "output_root": "outputs/tanks_and_temples/Ignatius/research/B2_PGSR",
    "experiment": "ignatius-pgsr", "seed": 42, "iterations": 30000,
    "resolution": 1, "sh_degree": 3, "white_background": False,
    "device": "cuda:0", "allow_cpu_fallback": False,
    "train_view_limit": None, "initial_point_limit": None,
    "cpu_camera_cache": 16, "ncc_scale": 1.0,
    "multi_view_num": 8, "multi_view_max_angle": 30.,
    "multi_view_min_dis": .01, "multi_view_max_dis": 1.5,
    "exposure_from_iter": 1000, "trim_interval": 1000,
    "checkpoint_interval": 1000, "log_interval": 100,
    "gradient_check_interval": 100, "torch_reserved_budget_gib": 12.,
    "session_seconds": 18000, "normal_epsilon": 1e-8,
    "depth_epsilon": 1e-6, "ncc_variance_epsilon": 1e-8,
    "optimizer": {
        "position_lr_init": .00016, "position_lr_final": .0000016,
        "position_lr_delay_mult": .01, "position_lr_max_steps": 30000,
        "feature_lr": .0025, "opacity_lr": .05, "scaling_lr": .005,
        "rotation_lr": .001, "percent_dense": .001, "lambda_dssim": .2,
        "densification_interval": 100, "opacity_reset_interval": 3000,
        "densify_from_iter": 500, "densify_until_iter": 15000,
        "densify_grad_threshold": .0002, "scale_loss_weight": 100.,
        "wo_image_weight": False, "single_view_weight": .015,
        "single_view_weight_from_iter": 7000, "use_virtul_cam": False,
        "virtul_cam_prob": .5, "use_multi_view_trim": True,
        "multi_view_ncc_weight": .15, "multi_view_geo_weight": .03,
        "multi_view_weight_from_iter": 7000, "multi_view_patch_size": 3,
        "multi_view_sample_num": 102400, "multi_view_pixel_noise_th": 1.,
        "wo_use_geo_occ_aware": False, "opacity_cull_threshold": .05,
        "densify_abs_grad_threshold": .00015,
        "abs_split_radii2D_threshold": 20, "max_abs_split_points": 50000,
        "max_all_points": 6000000, "exposure_compensation": True,
        "random_background": False,
    },
}


def load(path: Path) -> dict:
    config = deepcopy(DEFAULTS)
    supplied = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if set(supplied) - set(config):
        raise ValueError(f"Unknown PGSR keys: {set(supplied) - set(config)}")
    for key, value in supplied.items():
        if key == "optimizer":
            if set(value) - set(config[key]):
                raise ValueError("Unknown native optimization option")
            config[key].update(value)
        else:
            config[key] = value
    if config["device"] != "cuda:0" or config["allow_cpu_fallback"]:
        raise ValueError("PGSR requires cuda:0; CPU fallback is prohibited")
    if config["resolution"] not in (1, 2, 4, 8) or config["iterations"] < 1:
        raise ValueError("Invalid resolution/iteration profile")
    if config["optimizer"]["use_virtul_cam"]:
        raise ValueError("This audited baseline uses real train-only neighbors; virtual cameras disabled")
    if config["optimizer"]["single_view_weight_from_iter"] > config["optimizer"]["multi_view_weight_from_iter"]:
        raise ValueError("Multi-view geometry requires the reference plane outputs to be active")
    output = (ROOT/config["output_root"]).resolve()
    if not output.is_relative_to((ROOT/"outputs").resolve()) or "B2_PGSR" not in output.parts:
        raise ValueError("PGSR output must remain under outputs in its own B2_PGSR tree")
    return config
