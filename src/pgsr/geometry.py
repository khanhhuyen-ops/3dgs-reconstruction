"""Pinned PGSR RGB/plane/normal/roundtrip/LNCC objectives with explicit invalid masks.

Valid-sample equations/reductions follow de24f1a/train.py. Guards exclude undefined
projections, unsupported patches and empty/constant cases; no Ours terms are added.
"""
import numpy as np
import torch
import torch.nn.functional as F


def valid_geometry(package, camera, epsilon=1e-6):
    depth = package["plane_depth"].squeeze()
    normal = package["rendered_normal"]
    rays = camera.get_rays()
    denominator = -(normal.permute(1, 2, 0)*rays).sum(-1)
    return (torch.isfinite(depth) & (depth > epsilon)
            & torch.isfinite(normal).all(0) & (normal.norm(dim=0) > epsilon)
            & (denominator.abs() > epsilon)).detach()


def edge_weight(image):
    """Native normalized central-difference weight, with a constant-gradient guard."""
    dx = (image[:, 1:-1, 2:]-image[:, 1:-1, :-2]).abs().mean(0)
    dy = (image[:, :-2, 1:-1]-image[:, 2:, 1:-1]).abs().mean(0)
    gradient = torch.maximum(dx, dy)
    span = gradient.max()-gradient.min()
    normalized = (gradient-gradient.min())/span.clamp_min(1e-12)
    normalized = F.pad(normalized, (1, 1, 1, 1), value=1.)
    return (1-normalized).clamp(0, 1).detach().square()


def relative_pose(reference, neighbor):
    r = neighbor.world_view_transform[:3, :3].T @ reference.world_view_transform[:3, :3]
    t = -r @ reference.world_view_transform[3, :3] + neighbor.world_view_transform[3, :3]
    return r, t


def plane_homography(reference, neighbor, normal, distance):
    r, t = relative_pose(reference, neighbor)
    h = r[None]-t[None, :, None]@normal[:, None, :]/distance[:, None, None]
    return neighbor.get_k(neighbor.ncc_scale)[None] @ h @ reference.get_inv_k(reference.ncc_scale)


def warp_patch(homography, pixels, epsilon=1e-8):
    homogeneous = torch.cat([pixels, torch.ones_like(pixels[..., :1])], dim=-1)
    transformed = homogeneous @ homography.transpose(-1, -2)
    denominator = transformed[..., 2:]
    valid = torch.isfinite(transformed).all(-1) & (denominator.squeeze(-1).abs() > epsilon)
    divisor = torch.where(valid[..., None], denominator, 1)
    return transformed[..., :2]/divisor, valid


def objectives(model, app, reference, neighbor, package, neighbor_package,
               target, gray, neighbor_gray, options, config, iteration):
    from utils.loss_utils import l1_loss, ssim, lncc
    from utils.graphics_utils import patch_offsets

    if not target.is_cuda or not package["render"].is_cuda:
        raise RuntimeError("All PGSR loss computation requires CUDA")
    zero = model.get_xyz.sum()*0
    image = package["render"]
    structural = 1-ssim(image, target)
    exposure_used = "app_image" in package and bool(structural < .5)
    rgb = (1-options.lambda_dssim)*l1_loss(package["app_image"] if exposure_used else image, target)
    rgb = rgb + options.lambda_dssim*structural
    visible = package["visibility_filter"]
    flat = model.get_scaling[visible].sort(dim=-1).values[:, 0].mean() if visible.any() else zero
    single, geometric, photometric = zero, zero, zero
    counts = {"geometry_valid_pixels": 0, "multi_valid_pixels": 0,
              "ncc_candidate_patches": 0, "ncc_valid_patches": 0}
    if iteration > options.single_view_weight_from_iter:
        mask = valid_geometry(package, reference, config["depth_epsilon"])
        # A finite-difference normal needs every up/down/left/right sample valid.
        support = torch.zeros_like(mask)
        support[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[2:, 1:-1]
                                   & mask[:-2, 1:-1] & mask[1:-1, 2:] & mask[1:-1, :-2])
        difference = (package["depth_normal"]-package["rendered_normal"]).abs().sum(0)
        difference = torch.where(support, difference, torch.zeros_like(difference))
        weight = torch.ones_like(difference) if options.wo_image_weight else edge_weight(target)
        # Retain native H*W reduction rather than renormalizing by valid pixels.
        single = (weight*difference).mean()
        counts["geometry_valid_pixels"] = int(support.sum())
    if iteration > options.multi_view_weight_from_iter and neighbor is not None:
        mask_ref = valid_geometry(package, reference, config["depth_epsilon"])
        mask_near = valid_geometry(neighbor_package, neighbor, config["depth_epsilon"])
        depth = torch.where(mask_ref, package["plane_depth"].squeeze(), 0)
        height, width = depth.shape
        yy, xx = torch.meshgrid(torch.arange(height, device=depth.device),
                                torch.arange(width, device=depth.device), indexing="ij")
        pixels = torch.stack([xx, yy], dim=-1).reshape(-1, 2).float()
        world = model.get_points_from_depth(reference, depth)
        projected = world @ neighbor.world_view_transform[:3, :3] + neighbor.world_view_transform[3, :3]
        z = projected[:, 2]
        safe_z = torch.where(z.abs()>config["depth_epsilon"], z, torch.ones_like(z))
        xy = torch.stack([projected[:, 0]*neighbor.Fx/safe_z+neighbor.Cx,
                          projected[:, 1]*neighbor.Fy/safe_z+neighbor.Cy], -1)
        in_image = ((xy[:, 0] > 0) & (xy[:, 0] < neighbor.image_width)
                    & (xy[:, 1] > 0) & (xy[:, 1] < neighbor.image_height) & (z > .1))
        grid = xy.clone()
        grid[:, 0] = 2*grid[:, 0]/(neighbor.image_width-1)-1
        grid[:, 1] = 2*grid[:, 1]/(neighbor.image_height-1)-1
        grid = torch.nan_to_num(grid).view(1, -1, 1, 2)
        near_depth = torch.where(mask_near, neighbor_package["plane_depth"].squeeze(), 0)
        sampled = F.grid_sample(near_depth[None, None], grid, padding_mode="border", align_corners=True).reshape(-1)
        sampled_valid = F.grid_sample(mask_near.float()[None, None], grid,
                                      mode="nearest", align_corners=True).reshape(-1) > .5
        reconstructed = projected/safe_z[:, None]*sampled[:, None]
        r = torch.as_tensor(neighbor.R, dtype=projected.dtype, device=projected.device)
        t = torch.as_tensor(neighbor.T, dtype=projected.dtype, device=projected.device)
        world_back = (reconstructed-t)@r.T
        back = world_back @ reference.world_view_transform[:3, :3] + reference.world_view_transform[3, :3]
        safe_back = torch.where(back[:, 2].abs()>config["depth_epsilon"], back[:, 2], 1)
        reprojection = torch.stack([back[:, 0]*reference.Fx/safe_back+reference.Cx,
                                    back[:, 1]*reference.Fy/safe_back+reference.Cy], -1)
        noise = (reprojection-pixels).norm(dim=-1)
        valid = (mask_ref.reshape(-1) & in_image & sampled_valid & (sampled>config["depth_epsilon"])
                 & (back[:, 2]>config["depth_epsilon"]) & torch.isfinite(noise))
        if not options.wo_use_geo_occ_aware:
            valid = valid & (noise < options.multi_view_pixel_noise_th)
            weights = torch.exp(-noise).detach()
        else:
            weights = torch.ones_like(noise)
        weights = torch.where(valid, weights, 0)
        counts["multi_valid_pixels"] = int(valid.sum())
        if valid.any():
            geometric = (weights[valid]*noise[valid]).mean()
            with torch.no_grad():
                indices = torch.nonzero(valid, as_tuple=True)[0]
                if len(indices) > options.multi_view_sample_num:
                    chosen = np.random.choice(len(indices), options.multi_view_sample_num, replace=False)
                    indices = indices[torch.as_tensor(chosen, device=indices.device)]
                offsets = patch_offsets(options.multi_view_patch_size, pixels.device).float()
                patches = pixels[indices, None]/reference.ncc_scale + offsets
            distance = package["rendered_distance"].reshape(-1)[indices]
            normal = package["rendered_normal"].permute(1, 2, 0).reshape(-1, 3)[indices]
            distance_valid = torch.isfinite(distance) & (distance > config["depth_epsilon"])
            safe_distance = torch.where(distance_valid, distance, 1)
            homography = plane_homography(reference, neighbor, normal, safe_distance)
            warped, warp_valid = warp_patch(homography, patches)
            h, w = gray.shape[-2:]
            hn, wn = neighbor_gray.shape[-2:]
            patch_valid = (distance_valid & warp_valid.all(1) & torch.isfinite(warped).all((1, 2))
                           & (patches[..., 0]>=0).all(1) & (patches[..., 0]<=w-1).all(1)
                           & (patches[..., 1]>=0).all(1) & (patches[..., 1]<=h-1).all(1)
                           & (warped[..., 0]>=0).all(1) & (warped[..., 0]<=wn-1).all(1)
                           & (warped[..., 1]>=0).all(1) & (warped[..., 1]<=hn-1).all(1))
            counts["ncc_candidate_patches"] = int(patch_valid.sum())
            if patch_valid.any():
                patches, warped = patches[patch_valid].clone(), warped[patch_valid].clone()
                patches[..., 0] = 2*patches[..., 0]/(w-1)-1
                patches[..., 1] = 2*patches[..., 1]/(h-1)-1
                warped[..., 0] = 2*warped[..., 0]/(wn-1)-1
                warped[..., 1] = 2*warped[..., 1]/(hn-1)-1
                size = (options.multi_view_patch_size*2+1)**2
                ref_values = F.grid_sample(gray[None], patches.reshape(1, -1, 1, 2), align_corners=True).reshape(-1, size)
                near_values = F.grid_sample(neighbor_gray[None], warped.reshape(1, -1, 1, 2), align_corners=True).reshape(-1, size)
                ncc, ncc_mask = lncc(ref_values, near_values)
                variance = (ref_values.var(1, unbiased=False)>config["ncc_variance_epsilon"]) & (near_values.var(1, unbiased=False)>config["ncc_variance_epsilon"])
                good = ncc_mask.reshape(-1) & variance & torch.isfinite(ncc.reshape(-1))
                counts["ncc_valid_patches"] = int(good.sum())
                if good.any():
                    photometric = (ncc.reshape(-1)*weights[indices][patch_valid])[good].mean()
    losses = {"rgb": rgb, "flatten": flat, "single": single,
              "multi_geo": geometric, "multi_ncc": photometric}
    coefficients = {"rgb": 1., "flatten": options.scale_loss_weight,
                    "single": options.single_view_weight,
                    "multi_geo": options.multi_view_geo_weight,
                    "multi_ncc": options.multi_view_ncc_weight}
    total = sum(losses[k]*coefficients[k] for k in losses)
    return total, losses, coefficients, {**counts, "exposure_gate_used": exposure_used}
