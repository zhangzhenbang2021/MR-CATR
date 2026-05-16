import math
import os
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from sgm.inference.helpers import embed_watermark
from sgm.util import append_dims, default, instantiate_from_config
from torchvision.transforms import ToTensor
from tqdm import tqdm


def _dot(a, b, eps=1e-8):
    return torch.sum(a.float() * b.float())


def _norm(a, eps=1e-8):
    return torch.sqrt(torch.sum(a.float() * a.float()) + eps)


def _normalize(v, eps=1e-8):
    return v / (_norm(v, eps=eps) + eps)


def _cos(a, b, eps=1e-8):
    return _dot(a, b, eps=eps) / (_norm(a, eps=eps) * _norm(b, eps=eps) + eps)


def _channel_norm_map(v: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.sum(v.float() * v.float(), dim=1) + 1e-8)


def _channel_cos_map(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    num = torch.sum(a.float() * b.float(), dim=1)
    den = torch.sqrt(torch.sum(a.float() * a.float(), dim=1) + eps)
    den = den * torch.sqrt(torch.sum(b.float() * b.float(), dim=1) + eps)
    return num / (den + eps)


def _smooth_map(x: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    if kernel <= 1:
        return x.float()
    y = x.float().unsqueeze(1)
    y = F.avg_pool2d(y, kernel_size=kernel, stride=1, padding=kernel // 2)
    return y[:, 0]


def _quantile_t(x: torch.Tensor, q: float) -> torch.Tensor:
    flat = x.reshape(-1).float()
    return torch.quantile(flat, q)


def _rescale01(x: torch.Tensor, q: float = 0.95, eps: float = 1e-8) -> torch.Tensor:
    x = x.float()
    vmax = _quantile_t(x, q)
    return torch.clamp(x / (vmax + eps), 0.0, 1.0)


def motion_axis_from_denoised(denoised, eps=1e-8):
    res = denoised[1:] - denoised[:-1]
    m = torch.mean(res, dim=0)
    m = m.unsqueeze(0).repeat(denoised.shape[0], 1, 1, 1)
    return _normalize(m, eps=eps)


def consensus_axis(m_start, m_end, kappa=0.2, eps=1e-8):
    s_prior = _cos(m_start, m_end, eps=eps)
    if s_prior >= 0:
        m = _normalize((1 - kappa) * m_start + kappa * m_end, eps=eps)
    else:
        m = m_start
    return m, s_prior


def project_update(d_bwd, m, eps=1e-8):
    coeff = _dot(d_bwd, m, eps=eps)
    return coeff * m


def rho_schedule(sigma, sigma_T, p=2.0):
    return 1.0 - (sigma / sigma_T).clamp(min=0.0, max=1.0) ** p


def conflict_aware_fuse_released(d_bwd, d_bwd_aligned, sigma, sigma_T, p=2.0, eps=1e-8):
    s = _cos(d_bwd, d_bwd_aligned, eps=eps)
    s = torch.clamp(s, min=0.0)
    rho = rho_schedule(sigma, sigma_T, p=p)
    beta = rho * s
    d = (1 - beta) * d_bwd + beta * d_bwd_aligned
    return d, beta, s, rho


def _save_trace(trace: dict, trace_path: str):
    Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(trace, trace_path)


def shared_region_mask_from_denoised(
    denoised_start_rev: torch.Tensor,
    denoised_end: torch.Tensor,
    motion_score_map: torch.Tensor,
    smooth_kernel: int = 7,
    thresh_quantile: float = 0.72,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Soft shared-region mask for paper-style alignment diagnostics.
    A region is considered more shared when the start-conditioned and end-conditioned
    denoised predictions agree locally and have similar magnitudes.
    """
    local_agree = _smooth_map(_channel_cos_map(denoised_start_rev, denoised_end), kernel=smooth_kernel)
    local_agree01 = torch.clamp((local_agree + 1.0) * 0.5, 0.0, 1.0)

    start_norm = _smooth_map(_channel_norm_map(denoised_start_rev), kernel=smooth_kernel)
    end_norm = _smooth_map(_channel_norm_map(denoised_end), kernel=smooth_kernel)
    mag_balance = torch.minimum(start_norm, end_norm) / (torch.maximum(start_norm, end_norm) + eps)

    motion01 = _rescale01(_smooth_map(motion_score_map, kernel=smooth_kernel), q=0.95)
    shared_score = local_agree01 * mag_balance * (0.25 + 0.75 * motion01)
    thr = _quantile_t(shared_score, thresh_quantile)
    vmax = torch.max(shared_score)
    mask = torch.clamp((shared_score - thr) / (vmax - thr + eps), 0.0, 1.0)
    mask = _smooth_map(mask, kernel=smooth_kernel)

    debug = {
        "local_agree": local_agree,
        "mag_balance": mag_balance,
        "shared_score": shared_score,
        "motion01": motion01,
    }
    return mask, debug


def sample(
    input_start_path: str = "",
    input_end_path: str = "",
    num_frames: Optional[int] = None,
    num_steps: Optional[int] = None,
    version: str = "svd_xt",
    fps_id: int = 10,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 4,
    device: str = "cuda",
    cfg_scale: float = 1.0,
    cfg_scale_flip: float = 1.0,
    output_folder: Optional[str] = None,
    verbose: Optional[bool] = False,
    save_alignment_debug: bool = True,
    debug_dir: str = "alignment_debug",
    probe_stride: int = 1,
    probe_max_steps: int = 5,
    map_smooth_kernel: int = 5,
    shared_mask_kernel: int = 7,
    shared_mask_quantile: float = 0.72,
    use_shared_mask_in_sampler: bool = False,
):
    if version == "svd_xt":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 25)
        model_config = "svd/scripts/sampling/configs/svd_xt.yaml"
    else:
        raise ValueError(f"Version {version} does not exist.")

    model, filter = load_model(model_config, device, num_frames, num_steps, verbose)
    torch.manual_seed(seed)

    with Image.open(input_start_path) as image:
        input_image = image.convert("RGB")
        input_image = input_image.resize((1024, 576))
        w, h = input_image.size
        if h % 64 != 0 or w % 64 != 0:
            width, height = map(lambda x: x - x % 64, (w, h))
            input_image = input_image.resize((width, height))
            print(f"WARNING: image size {h}x{w} is not divisible by 64. Resizing to {height}x{width}.")

    image = ToTensor()(input_image)
    image = image * 2.0 - 1.0
    image = image.unsqueeze(0).to(device).to(torch.float16)
    latent = model.encode_first_stage(image)

    input_image_end = Image.open(input_end_path).convert("RGB").resize((1024, 576))
    image_end = ToTensor()(input_image_end)
    image_end = image_end * 2.0 - 1.0
    image_end = image_end.unsqueeze(0).to(device).to(torch.float16)
    latent_end = model.encode_first_stage(image_end)

    H, W = image.shape[2:]
    F_down = 8
    C = 4
    shape = (num_frames, C, H // F_down, W // F_down)

    value_dict = {
        "cond_frames_without_noise": image,
        "motion_bucket_id": motion_bucket_id,
        "fps_id": fps_id,
        "cond_aug": cond_aug,
        "cond_frames": image + cond_aug * torch.randn_like(image),
    }
    value_dict_end = {
        "cond_frames_without_noise": image_end,
        "motion_bucket_id": motion_bucket_id,
        "fps_id": fps_id,
        "cond_aug": cond_aug,
        "cond_frames": image_end + cond_aug * torch.randn_like(image_end),
    }

    case_name = Path(input_start_path).parent.name or Path(input_start_path).stem
    trace = {
        "case_name": case_name,
        "num_frames": int(num_frames),
        "records": [],
        "sampler_note": "released sampler unchanged by default; heatmap diagnostics add shared-region masking and best-step selection",
        "use_shared_mask_in_sampler": bool(use_shared_mask_in_sampler),
    }
    probed_count = 0

    with torch.no_grad():
        with torch.autocast(device):
            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner),
                value_dict,
                [1, num_frames],
                T=num_frames,
                device=device,
            )
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=["cond_frames", "cond_frames_without_noise"],
            )
            for k in ["crossattn", "concat"]:
                uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

            batch_end, batch_uc_end = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner),
                value_dict_end,
                [1, num_frames],
                T=num_frames,
                device=device,
            )
            c_end, uc_end = model.conditioner.get_unconditional_conditioning(
                batch_end,
                batch_uc=batch_uc_end,
                force_uc_zero_embeddings=["cond_frames", "cond_frames_without_noise"],
            )
            for k in ["crossattn", "concat"]:
                uc_end[k] = repeat(uc_end[k], "b ... -> b t ...", t=num_frames)
                uc_end[k] = rearrange(uc_end[k], "b t ... -> (b t) ...", t=num_frames)
                c_end[k] = repeat(c_end[k], "b ... -> b t ...", t=num_frames)
                c_end[k] = rearrange(c_end[k], "b t ... -> (b t) ...", t=num_frames)

            randn = torch.randn(shape, device=device)
            additional_model_inputs = {
                "image_only_indicator": torch.zeros(2, num_frames).to(device),
                "num_video_frames": batch["num_video_frames"],
            }

            def denoiser(x, sigma, c_local, uc_local):
                c_out = {}
                for k in c_local:
                    if k in ["vector", "crossattn", "concat"]:
                        c_out[k] = torch.cat((uc_local[k], c_local[k]), 0)
                    else:
                        assert c_local[k] == uc_local[k]
                        c_out[k] = c_local[k]
                denoiser_input = torch.cat([x] * 2)
                denoiser_sigma = torch.cat([sigma] * 2)
                sigma_shape = denoiser_sigma.shape
                denoiser_sigma = append_dims(denoiser_sigma, x.ndim)
                c_skip = 1.0 / (denoiser_sigma ** 2 + 1.0)
                c_out_scale = -denoiser_sigma / (denoiser_sigma ** 2 + 1.0) ** 0.5
                c_in = 1.0 / (denoiser_sigma ** 2 + 1.0) ** 0.5
                c_noise = 0.25 * denoiser_sigma.log()
                c_noise = c_noise.reshape(sigma_shape)
                denoised = model.model(
                    denoiser_input * c_in,
                    c_noise,
                    c_out,
                    **additional_model_inputs,
                ) * c_out_scale + denoiser_input * c_skip
                x_u, x_c = denoised.chunk(2)
                return x_u, x_c

            def CFG(x_u, x_c, scale):
                x_u_local = rearrange(x_u, "(b t) ... -> b t ...", t=num_frames)
                x_c_local = rearrange(x_c, "(b t) ... -> b t ...", t=num_frames)
                scale_tensor = torch.linspace(scale, scale, steps=num_frames).unsqueeze(0)
                scale_tensor = repeat(scale_tensor, "1 t -> b t", b=x_u_local.shape[0])
                scale_tensor = append_dims(scale_tensor, x_u_local.ndim).to(x_u_local.device)
                return rearrange(x_u_local + scale_tensor * (x_c_local - x_u_local), "b t ... -> (b t) ...")

            def masking(x, index):
                mask = torch.zeros_like(x)
                mask[index, :, :, :] = 1
                return x * mask

            def CG(A, b, x, n_inner=5, eps=1e-5):
                r = b - A(x)
                p = r.clone()
                rsold = torch.sum(r * r, dim=[0, 1, 2, 3], keepdim=True)
                for _ in range(n_inner):
                    Ap = A(p)
                    a = rsold / torch.sum(p * Ap, dim=[0, 1, 2, 3], keepdim=True)
                    x = x + a * p
                    r = r - a * Ap
                    rsnew = torch.sum(r * r, dim=[0, 1, 2, 3], keepdim=True)
                    if torch.abs(torch.sqrt(rsnew)) < eps:
                        break
                    p = r + (rsnew / rsold) * p
                    rsold = rsnew
                return x

            def DDS(x, n_inner, latent_target):
                measurement = torch.zeros_like(x)
                measurement[-1, :, :, :] = latent_target
                A = lambda z: masking(z, -1)
                AT = lambda z: masking(z, -1)
                def Acg(z):
                    return AT(A(z))
                bcg = AT(measurement)
                return CG(Acg, bcg, x, n_inner=n_inner)

            def local_forward_reference_at_xbase(x_base, sigma_hat_local, next_sigma_local):
                x_same = torch.flip(x_base, dims=[0])
                x_u_same, x_c_same = denoiser(x_same, sigma_hat_local, cond, uc)
                denoised_same = CFG(x_u_same, x_c_same, scale=cfg_scale)
                denoised_hat_same = DDS(denoised_same, n_inner=5, latent_target=latent_end)
                d_same = (x_same - x_u_same) / append_dims(sigma_hat_local, x_same.ndim)
                dt_same = append_dims(next_sigma_local, x_same.ndim)
                x_same_next = denoised_hat_same + d_same * dt_same
                d_fwd_same = x_same_next - x_same
                return torch.flip(d_fwd_same, dims=[0])

            x, s_in, sigmas, num_sigmas, cond, uc = model.sampler.prepare_sampling_loop(randn, c, uc, num_steps)

            for i in tqdm(model.sampler.get_sigma_gen(num_sigmas), total=num_sigmas - 1):
                gamma = (
                    min(model.sampler.s_churn / (num_sigmas - 1), 2 ** 0.5 - 1)
                    if model.sampler.s_tmin <= sigmas[i] <= model.sampler.s_tmax
                    else 0.0
                )
                sigma = s_in * sigmas[i]
                next_sigma = s_in * sigmas[i + 1]
                sigma_hat = sigma * (gamma + 1.0)

                if gamma > 0:
                    eps = torch.randn_like(x) * model.sampler.s_noise
                    x = x + eps * append_dims(sigma_hat ** 2 - sigma ** 2, x.ndim) ** 0.5

                if i < 0.2 * num_steps:
                    x_u, x_c = denoiser(x, sigma_hat, cond, uc)
                    denoised = CFG(x_u, x_c, scale=cfg_scale)
                    denoised_start_for_axis = denoised.detach()

                    denoised_hat = DDS(denoised, n_inner=5, latent_target=latent_end)
                    d = (x - x_u) / append_dims(sigma_hat, x.ndim)
                    dt = append_dims(next_sigma, x.ndim)
                    x = denoised_hat + d * dt

                    eps = torch.randn_like(x) * model.sampler.s_noise
                    x = x + eps * append_dims(sigma_hat ** 2 - next_sigma ** 2, x.ndim) ** 0.5

                    x = torch.flip(x, dims=[0])
                    x_base = x

                    x_u_end, x_c_end = denoiser(x, sigma_hat, c_end, uc_end)
                    denoised_end = CFG(x_u_end, x_c_end, scale=cfg_scale_flip)

                    start_rev = torch.flip(denoised_start_for_axis, dims=[0])
                    m_start = motion_axis_from_denoised(start_rev)
                    m_end = motion_axis_from_denoised(denoised_end.detach())
                    m, s_prior = consensus_axis(m_start, m_end, kappa=0.2)

                    denoised_hat = DDS(denoised_end, n_inner=5, latent_target=latent)
                    d_raw = (x_base - x_u_end) / append_dims(sigma_hat, x.ndim)
                    dt = append_dims(next_sigma, x.ndim)
                    x_raw_next = denoised_hat + d_raw * dt
                    d_bwd = x_raw_next - x_base
                    d_bwd_aligned = project_update(d_bwd, m)

                    d_fwd_ref = local_forward_reference_at_xbase(x_base, sigma_hat, next_sigma)
                    raw_vs_fwd = _smooth_map(_channel_cos_map(d_bwd, d_fwd_ref), kernel=map_smooth_kernel)
                    aligned_vs_fwd = _smooth_map(_channel_cos_map(d_bwd_aligned, d_fwd_ref), kernel=map_smooth_kernel)

                    bwd_norm = _channel_norm_map(d_bwd)
                    aligned_norm = _channel_norm_map(d_bwd_aligned)
                    fwd_ref_norm = _channel_norm_map(d_fwd_ref)
                    motion_score_map = torch.maximum(fwd_ref_norm, torch.maximum(aligned_norm, bwd_norm))

                    shared_mask, shared_debug = shared_region_mask_from_denoised(
                        start_rev,
                        denoised_end.detach(),
                        motion_score_map,
                        smooth_kernel=shared_mask_kernel,
                        thresh_quantile=shared_mask_quantile,
                    )
                    shared_mask_4d = shared_mask.unsqueeze(1)
                    d_bwd_masked = shared_mask_4d * d_bwd_aligned + (1.0 - shared_mask_4d) * d_bwd
                    d_orth_masked = d_bwd - d_bwd_masked

                    masked_vs_fwd = _smooth_map(_channel_cos_map(d_bwd_masked, d_fwd_ref), kernel=map_smooth_kernel)
                    delta_masked_vs_fwd = masked_vs_fwd - raw_vs_fwd

                    candidate_for_fuse = d_bwd_masked if use_shared_mask_in_sampler else d_bwd_aligned
                    d_fused, beta, s_agree, rho = conflict_aware_fuse_released(
                        d_bwd,
                        candidate_for_fuse,
                        sigmas[i],
                        sigmas[0],
                        p=2.0,
                    )

                    should_probe = (
                        save_alignment_debug
                        and (i % max(probe_stride, 1) == 0)
                        and probed_count < max(probe_max_steps, 1)
                    )
                    if should_probe:
                        proj_ratio_map = _smooth_map(
                            _channel_norm_map(d_bwd_masked - d_bwd) / (_channel_norm_map(d_bwd) + 1e-8),
                            kernel=map_smooth_kernel,
                        )
                        gain_focus_map = _smooth_map(
                            torch.relu(delta_masked_vs_fwd)
                            * (0.25 + 0.75 * _rescale01(motion_score_map, q=0.95))
                            * (0.25 + 0.75 * shared_mask),
                            kernel=map_smooth_kernel,
                        )

                        record = {
                            "step_idx": int(i),
                            "sigma": float(sigmas[i]),
                            "rho": float(rho),
                            "beta": float(beta),
                            "s_prior": float(s_prior),
                            "s_agree": float(s_agree),
                            "raw_vs_fwd_map": raw_vs_fwd.detach().cpu().to(torch.float16),
                            "aligned_vs_fwd_map": aligned_vs_fwd.detach().cpu().to(torch.float16),
                            "masked_vs_fwd_map": masked_vs_fwd.detach().cpu().to(torch.float16),
                            "delta_masked_vs_fwd_map": delta_masked_vs_fwd.detach().cpu().to(torch.float16),
                            "shared_mask_map": shared_mask.detach().cpu().to(torch.float16),
                            "shared_local_agree_map": shared_debug["local_agree"].detach().cpu().to(torch.float16),
                            "shared_score_map": shared_debug["shared_score"].detach().cpu().to(torch.float16),
                            "motion_score_map": motion_score_map.detach().cpu().to(torch.float16),
                            "proj_ratio_map": proj_ratio_map.detach().cpu().to(torch.float16),
                            "gain_focus_map": gain_focus_map.detach().cpu().to(torch.float16),
                        }
                        trace["records"].append(record)
                        probed_count += 1

                    x = x_base + d_fused
                    x = torch.flip(x, dims=[0])
                else:
                    x_u, x_c = denoiser(x, sigma_hat, cond, uc)
                    denoised = CFG(x_u, x_c, scale=cfg_scale)
                    denoised_hat = DDS(denoised, n_inner=5, latent_target=latent_end)
                    d = (x - x_u) / append_dims(sigma_hat, x.ndim)
                    dt = append_dims(next_sigma, x.ndim)
                    x = denoised_hat + d * dt

                    eps = torch.randn_like(x) * model.sampler.s_noise
                    x = x + eps * append_dims(sigma_hat ** 2 - next_sigma ** 2, x.ndim) ** 0.5
                    x = torch.flip(x, dims=[0])
                    x_u, x_c = denoiser(x, sigma_hat, c_end, uc_end)
                    denoised = CFG(x_u, x_c, scale=cfg_scale_flip)
                    denoised_hat = DDS(denoised, n_inner=5, latent_target=latent)
                    d = (x - x_u) / append_dims(sigma_hat, x.ndim)
                    dt = append_dims(next_sigma, x.ndim)
                    x = denoised_hat + d * dt
                    x = torch.flip(x, dims=[0])

            samples_z = x
            model.en_and_decode_n_samples_a_time = decoding_t
            model = model.to(torch.float32)
            samples_x = model.decode_first_stage(samples_z)
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

            os.makedirs(output_folder, exist_ok=True)
            base_count = len(glob(os.path.join(output_folder, "*.gif")))
            samples = embed_watermark(samples)
            if filter is not None:
                samples = filter(samples)
            vid = (rearrange(samples, "t c h w -> t h w c") * 255).cpu().numpy().astype(np.uint8)
            video_path = os.path.join(output_folder, f"{base_count:06d}.gif")
            images = [Image.fromarray(vid[i]) for i in range(vid.shape[0])]
            images[0].save(video_path, save_all=True, append_images=images[1:], duration=125, loop=0)

    if save_alignment_debug:
        trace["video_path"] = video_path
        trace_path = os.path.join(debug_dir, f"{case_name}_trace.pt")
        _save_trace(trace, trace_path)
        print(f"[alignment-heatmap-masked] saved trace to: {trace_path}")
        print(f"[alignment-heatmap-masked] generated gif: {video_path}")


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}
    for key in keys:
        if key == "fps_id":
            batch[key] = torch.tensor([value_dict["fps_id"]]).to(device).repeat(int(math.prod(N)))
        elif key == "motion_bucket_id":
            batch[key] = torch.tensor([value_dict["motion_bucket_id"]]).to(device).repeat(int(math.prod(N)))
        elif key == "cond_aug":
            batch[key] = repeat(torch.tensor([value_dict["cond_aug"]]).to(device), "1 -> b", b=math.prod(N))
        elif key in ["cond_frames", "cond_frames_without_noise"]:
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0])
        elif key in ["polars_rad", "azimuths_rad"]:
            batch[key] = torch.tensor(value_dict[key]).to(device).repeat(N[0])
        else:
            batch[key] = value_dict[key]
    if T is not None:
        batch["num_video_frames"] = T
    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_model(config: str, device: str, num_frames: int, num_steps: int, verbose: bool = False):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[0].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.verbose = verbose
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = num_frames
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    model = model.to(torch.float16)
    filter = None
    return model, filter


if __name__ == "__main__":
    Fire(sample)
