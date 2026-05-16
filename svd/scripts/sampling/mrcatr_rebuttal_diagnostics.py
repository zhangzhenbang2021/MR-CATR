import argparse
import csv
import json
import math
import os
import sys
from contextlib import nullcontext
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import append_dims, default, instantiate_from_config


# -----------------------------------------------------------------------------
# Geometry helpers copied from the main sampler so diagnostics match the paper
# -----------------------------------------------------------------------------

def _dot(a, b, eps: float = 1e-8):
    return torch.sum(a.float() * b.float())


def _norm(a, eps: float = 1e-8):
    return torch.sqrt(torch.sum(a.float() * a.float()) + eps)


def _normalize(v, eps: float = 1e-8):
    return v / (_norm(v, eps=eps) + eps)


def _cos(a, b, eps: float = 1e-8):
    return _dot(a, b, eps=eps) / (_norm(a, eps=eps) * _norm(b, eps=eps) + eps)


def motion_axis_from_denoised(denoised, eps: float = 1e-8):
    res = denoised[1:] - denoised[:-1]
    m = torch.mean(res, dim=0)
    m = m.unsqueeze(0).repeat(denoised.shape[0], 1, 1, 1)
    return _normalize(m, eps=eps)


def consensus_axis(m_start, m_end, kappa: float = 0.2, eps: float = 1e-8):
    s_prior = _cos(m_start, m_end, eps=eps)
    if s_prior >= 0:
        m = _normalize((1 - kappa) * m_start + kappa * m_end, eps=eps)
    else:
        m = m_start
    return m, s_prior


def project_update(d_bwd, m, eps: float = 1e-8):
    coeff = _dot(d_bwd, m, eps=eps)
    return coeff * m


def rho_schedule(sigma, sigma_T, p: float = 2.0):
    return 1.0 - (sigma / sigma_T).clamp(min=0.0, max=1.0) ** p


def conflict_aware_fuse(d_bwd, d_bwd_aligned, sigma, sigma_T, p: float = 2.0, eps: float = 1e-8):
    s = _cos(d_bwd, d_bwd_aligned, eps=eps)
    s = torch.clamp(s, min=0.0)
    rho = rho_schedule(sigma, sigma_T, p=p)
    beta = rho * s
    d = (1 - beta) * d_bwd + beta * d_bwd_aligned
    return d, beta, s, rho


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def device_type_from_device(device: str) -> str:
    return "cuda" if str(device).startswith("cuda") else "cpu"


def autocast_context(device: str):
    dtype = device_type_from_device(device)
    if dtype == "cuda":
        return torch.autocast(device_type="cuda")
    return nullcontext()


def parse_csv_list(text: Optional[str]) -> List[str]:
    if text is None:
        return []
    text = text.strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_snapshot_steps(text: str, num_steps: int, active_fraction: float) -> List[int]:
    text = text.strip().lower()
    if text == "auto":
        active_steps = max(1, int(math.ceil(active_fraction * num_steps)))
        if active_steps == 1:
            return [0]
        if active_steps == 2:
            return [0, 1]
        return sorted({0, active_steps // 2, active_steps - 1})
    steps = []
    for item in parse_csv_list(text):
        steps.append(int(item))
    return sorted(set(steps))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_metrics_csv(rows: List[Dict], path: Path):
    if not rows:
        raise ValueError(f"No rows to save into {path}")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tensor_scalar(x) -> float:
    if not torch.is_tensor(x):
        return float(x)

    t = x.detach().float().cpu()
    if t.numel() == 1:
        return float(t.item())

    t = t.reshape(-1)
    first = t[0]
    if torch.allclose(t, first.expand_as(t), atol=1e-6, rtol=1e-5):
        return float(first.item())

    return float(t.mean().item())

def choose_cases_from_defaults(data_root: Path, selected_case_names: List[str]) -> List[Dict[str, str]]:
    default_case_names = ["train", "bear", "boat", "car-turn"]
    if not selected_case_names or selected_case_names == ["all"]:
        case_names = default_case_names
    else:
        case_names = selected_case_names

    cases = []
    for case_name in case_names:
        case_dir = data_root / case_name
        cases.append(
            {
                "name": case_name,
                "start": str(case_dir / "00000.jpg"),
                "end": str(case_dir / "00024.jpg"),
            }
        )
    return cases


def load_case_file(case_file: Path) -> List[Dict[str, str]]:
    with open(case_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("case_file must be a JSON list of {'name', 'start', 'end'} objects.")
    required = {"name", "start", "end"}
    for idx, item in enumerate(data):
        if not required.issubset(item.keys()):
            missing = required - set(item.keys())
            raise ValueError(f"Case #{idx} is missing keys: {sorted(missing)}")
    return data


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
        elif key == "cond_frames" or key == "cond_frames_without_noise":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0])
        elif key == "polars_rad" or key == "azimuths_rad":
            batch[key] = torch.tensor(value_dict[key]).to(device).repeat(N[0])
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
    verbose: bool = False,
    with_filter: bool = True,
):
    config = OmegaConf.load(config)
    if device_type_from_device(device) == "cuda":
        config.model.params.conditioner_config.params.emb_models[0].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.verbose = verbose
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = num_frames

    model = instantiate_from_config(config.model).to(device).eval()
    model = model.to(torch.float16)

    filter_model = None
    if with_filter:
        filter_model = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter_model


# -----------------------------------------------------------------------------
# Main diagnostic logic
# -----------------------------------------------------------------------------

def build_conditionings(model, image, image_end, motion_bucket_id, fps_id, cond_aug, num_frames, device):
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
        uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
        uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
        c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
        c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

        uc_end[k] = repeat(uc_end[k], "b ... -> b t ...", t=num_frames)
        uc_end[k] = rearrange(uc_end[k], "b t ... -> (b t) ...", t=num_frames)
        c_end[k] = repeat(c_end[k], "b ... -> b t ...", t=num_frames)
        c_end[k] = rearrange(c_end[k], "b t ... -> (b t) ...", t=num_frames)

    return (c, uc), (c_end, uc_end), batch


def load_endpoint_images(model, input_start_path: str, input_end_path: str, device: str):
    with Image.open(input_start_path) as image_pil:
        input_image = image_pil.convert("RGB")
        input_image = input_image.resize((1024, 576))
        w, h = input_image.size
        if h % 64 != 0 or w % 64 != 0:
            width, height = map(lambda x: x - x % 64, (w, h))
            input_image = input_image.resize((width, height))
            print(
                f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. Resizing to {height}x{width}."
            )

    image = ToTensor()(input_image)
    image = image * 2.0 - 1.0
    image = image.unsqueeze(0).to(device).to(torch.float16)
    latent = model.encode_first_stage(image)

    input_image_end = Image.open(input_end_path).convert("RGB").resize((1024, 576))
    image_end = ToTensor()(input_image_end)
    image_end = image_end * 2.0 - 1.0
    image_end = image_end.unsqueeze(0).to(device).to(torch.float16)
    latent_end = model.encode_first_stage(image_end)

    return image, latent, image_end, latent_end


def sample_case(
    model,
    filter_model,
    case_name: str,
    input_start_path: str,
    input_end_path: str,
    output_root: Path,
    analysis_root: Path,
    num_frames: Optional[int],
    num_steps: Optional[int],
    version: str,
    fps_id: int,
    motion_bucket_id: int,
    cond_aug: float,
    seed: int,
    decoding_t: int,
    device: str,
    cfg_scale: float,
    cfg_scale_flip: float,
    verbose: bool,
    active_fraction: float,
    snapshot_steps: List[int],
    save_snapshots: bool,
    save_metrics: bool,
    save_video: bool,
    dds_cg_inner: int,
):
    if version == "svd_xt":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 25)
    else:
        raise ValueError(f"Version {version} does not exist.")

    torch.manual_seed(seed)

    if not os.path.exists(input_start_path):
        raise FileNotFoundError(f"Start image not found: {input_start_path}")
    if not os.path.exists(input_end_path):
        raise FileNotFoundError(f"End image not found: {input_end_path}")

    image, latent, image_end, latent_end = load_endpoint_images(model, input_start_path, input_end_path, device)

    H, W = image.shape[2:]
    assert image.shape[1] == 3
    F = 8
    C = 4
    shape = (num_frames, C, H // F, W // F)

    (c, uc), (c_end, uc_end), batch = build_conditionings(
        model=model,
        image=image,
        image_end=image_end,
        motion_bucket_id=motion_bucket_id,
        fps_id=fps_id,
        cond_aug=cond_aug,
        num_frames=num_frames,
        device=device,
    )

    randn = torch.randn(shape, device=device)

    additional_model_inputs = {
        "image_only_indicator": torch.zeros(2, num_frames).to(device),
        "num_video_frames": batch["num_video_frames"],
    }

    def denoiser(x, sigma, c, uc):
        c_out = {}
        for k in c:
            if k in ["vector", "crossattn", "concat"]:
                c_out[k] = torch.cat((uc[k], c[k]), 0)
            else:
                assert c[k] == uc[k]
                c_out[k] = c[k]
        denoiser_input = torch.cat([x] * 2)
        denoiser_sigma = torch.cat([sigma] * 2)
        denoiser_c = c_out
        sigma_shape = denoiser_sigma.shape
        denoiser_sigma_appended = append_dims(denoiser_sigma, x.ndim)
        c_skip = 1.0 / (denoiser_sigma_appended**2 + 1.0)
        c_out_coeff = -denoiser_sigma_appended / (denoiser_sigma_appended**2 + 1.0) ** 0.5
        c_in = 1.0 / (denoiser_sigma_appended**2 + 1.0) ** 0.5
        c_noise = 0.25 * denoiser_sigma.log()
        c_noise = c_noise.reshape(sigma_shape)
        denoised = model.model(denoiser_input * c_in, c_noise, denoiser_c, **additional_model_inputs) * c_out_coeff + denoiser_input * c_skip
        x_u, x_c = denoised.chunk(2)
        return x_u, x_c

    def cfg(x_u, x_c, scale):
        x_u = rearrange(x_u, "(b t) ... -> b t ...", t=num_frames)
        x_c = rearrange(x_c, "(b t) ... -> b t ...", t=num_frames)
        scale_tensor = torch.linspace(scale, scale, steps=num_frames).unsqueeze(0)
        scale_tensor = repeat(scale_tensor, "1 t -> b t", b=x_u.shape[0])
        scale_tensor = append_dims(scale_tensor, x_u.ndim).to(x_u.device)
        denoised = rearrange(x_u + scale_tensor * (x_c - x_u), "b t ... -> (b t) ...")
        return denoised

    def masking(x, index):
        mask = torch.zeros_like(x)
        mask[index, :, :, :] = 1
        return x * mask

    def cg(A, b, x, n_inner=5, eps=1e-5):
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

    def dds(x, n_inner, latent_target):
        measurement = torch.zeros_like(x)
        measurement[-1, :, :, :] = latent_target
        A = lambda z: masking(z, -1)
        AT = lambda z: masking(z, -1)

        def Acg(z):
            return AT(A(z))

        bcg = AT(measurement)
        return cg(Acg, bcg, x, n_inner=n_inner)

    x, s_in, sigmas, num_sigmas, cond, uc_loop = model.sampler.prepare_sampling_loop(randn, c, uc, num_steps)

    active_step_records: List[Dict] = []
    snapshot_steps_set = set(snapshot_steps)
    snapshots_dir = ensure_dir(analysis_root / case_name / "snapshots")
    output_case_dir = ensure_dir(output_root / case_name)

    meta = {
        "case_name": case_name,
        "input_start_path": input_start_path,
        "input_end_path": input_end_path,
        "num_frames": int(num_frames),
        "num_steps": int(num_steps),
        "active_fraction": float(active_fraction),
        "active_steps_requested": snapshot_steps,
        "save_snapshots": bool(save_snapshots),
        "save_metrics": bool(save_metrics),
        "save_video": bool(save_video),
        "seed": int(seed),
        "cfg_scale": float(cfg_scale),
        "cfg_scale_flip": float(cfg_scale_flip),
        "fps_id": int(fps_id),
        "motion_bucket_id": int(motion_bucket_id),
        "cond_aug": float(cond_aug),
        "dds_cg_inner": int(dds_cg_inner),
        "device": str(device),
        "version": str(version),
    }

    active_counter = 0
    total_effective_steps = num_sigmas - 1
    active_step_limit = int(math.ceil(active_fraction * num_steps))
    print(f"[{case_name}] total denoising steps in loop: {total_effective_steps}, active diagnostic steps: {active_step_limit}")

    with torch.no_grad():
        with autocast_context(device):
            for i in tqdm(model.sampler.get_sigma_gen(num_sigmas), total=num_sigmas - 1, desc=f"MR-CATR {case_name}"):
                gamma = (
                    min(model.sampler.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                    if model.sampler.s_tmin <= sigmas[i] <= model.sampler.s_tmax
                    else 0.0
                )
                sigma = s_in * sigmas[i]
                next_sigma = s_in * sigmas[i + 1]
                sigma_hat = sigma * (gamma + 1.0)

                if gamma > 0:
                    eps_noise = torch.randn_like(x) * model.sampler.s_noise
                    x = x + eps_noise * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

                if i < active_fraction * num_steps:
                    x_u, x_c = denoiser(x, sigma_hat, cond, uc_loop)
                    denoised = cfg(x_u, x_c, scale=cfg_scale)
                    denoised_start_for_axis = denoised.detach()

                    denoised_hat = dds(denoised, n_inner=dds_cg_inner, latent_target=latent_end)
                    d = (x - x_u) / append_dims(sigma_hat, x.ndim)
                    dt = append_dims(next_sigma, x.ndim)
                    x = denoised_hat + d * dt

                    eps_noise = torch.randn_like(x) * model.sampler.s_noise
                    x = x + eps_noise * append_dims(sigma_hat**2 - next_sigma**2, x.ndim) ** 0.5

                    x = torch.flip(x, dims=[0])
                    x_base = x

                    x_u_end, x_c_end = denoiser(x, sigma_hat, c_end, uc_end)
                    denoised_end = cfg(x_u_end, x_c_end, scale=cfg_scale_flip)

                    m_start = motion_axis_from_denoised(torch.flip(denoised_start_for_axis, dims=[0]))
                    m_end = motion_axis_from_denoised(denoised_end.detach())
                    m, s_prior = consensus_axis(m_start, m_end, kappa=0.2)

                    denoised_hat_end = dds(denoised_end, n_inner=dds_cg_inner, latent_target=latent)
                    d_raw = (x_base - x_u_end) / append_dims(sigma_hat, x.ndim)
                    x_raw_next = denoised_hat_end + d_raw * dt
                    d_bwd = x_raw_next - x_base
                    d_bwd_aligned = project_update(d_bwd, m)
                    d_perp = d_bwd - d_bwd_aligned

                    # Analysis-only forward auxiliary update from the SAME noisy state.
                    x_aux_forward = torch.flip(x_base, dims=[0])
                    x_u_aux, x_c_aux = denoiser(x_aux_forward, sigma_hat, cond, uc_loop)
                    denoised_aux = cfg(x_u_aux, x_c_aux, scale=cfg_scale)
                    denoised_hat_aux = dds(denoised_aux, n_inner=dds_cg_inner, latent_target=latent_end)
                    d_aux = (x_aux_forward - x_u_aux) / append_dims(sigma_hat, x.ndim)
                    x_aux_next = denoised_hat_aux + d_aux * dt
                    d_fwd_aux = torch.flip(x_aux_next - x_aux_forward, dims=[0])

                    d_fused, beta, s_agree, rho = conflict_aware_fuse(d_bwd, d_bwd_aligned, sigmas[i], sigmas[0], p=2.0)
                    x = x_base + d_fused

                    record = {
                        "case_name": case_name,
                        "loop_step": int(i),
                        "active_step_rank": int(active_counter),
                        "sigma": tensor_scalar(sigmas[i]),
                        "sigma_hat": tensor_scalar(sigma_hat),
                        "next_sigma": tensor_scalar(sigmas[i + 1]),
                        "beta": tensor_scalar(beta),
                        "s_agree": tensor_scalar(s_agree),
                        "rho": tensor_scalar(rho),
                        "s_prior": tensor_scalar(s_prior),
                        "cos_bwd_fwd_aux": tensor_scalar(_cos(d_bwd, d_fwd_aux)),
                        "cos_aligned_fwd_aux": tensor_scalar(_cos(d_bwd_aligned, d_fwd_aux)),
                        "cos_bwd_axis": tensor_scalar(_cos(d_bwd, m)),
                        "cos_aligned_axis": tensor_scalar(_cos(d_bwd_aligned, m)),
                        "norm_d_bwd": tensor_scalar(_norm(d_bwd)),
                        "norm_d_bwd_aligned": tensor_scalar(_norm(d_bwd_aligned)),
                        "norm_d_perp": tensor_scalar(_norm(d_perp)),
                        "norm_d_fwd_aux": tensor_scalar(_norm(d_fwd_aux)),
                        "norm_d_fused": tensor_scalar(_norm(d_fused)),
                        "axis_support_ratio": tensor_scalar(_norm(d_bwd_aligned) / (_norm(d_bwd) + 1e-8)),
                        "effective_correction_ratio": tensor_scalar(_norm(d_fused - d_bwd) / (_norm(d_bwd) + 1e-8)),
                        "cos_fused_axis": tensor_scalar(_cos(d_fused, m)),
                        "orth_ratio": tensor_scalar(_norm(d_perp) / (_norm(d_bwd) + 1e-8)),
                        "projection_coeff": tensor_scalar(_dot(d_bwd, m)),
                        "positive_before": int(tensor_scalar(_cos(d_bwd, d_fwd_aux)) > 0.0),
                        "positive_after": int(tensor_scalar(_cos(d_bwd_aligned, d_fwd_aux)) > 0.0),
                    }
                    active_step_records.append(record)

                    if save_snapshots and active_counter in snapshot_steps_set:
                        snapshot = {
                            "case_name": case_name,
                            "loop_step": int(i),
                            "active_step_rank": int(active_counter),
                            "sigma": tensor_scalar(sigmas[i]),
                            "sigma_hat": tensor_scalar(sigma_hat),
                            "next_sigma": tensor_scalar(sigmas[i + 1]),
                            "beta": tensor_scalar(beta),
                            "s_agree": tensor_scalar(s_agree),
                            "rho": tensor_scalar(rho),
                            "s_prior": tensor_scalar(s_prior),
                            "x_base": x_base.detach().cpu().to(torch.float16),
                            "d_bwd": d_bwd.detach().cpu().to(torch.float16),
                            "d_bwd_aligned": d_bwd_aligned.detach().cpu().to(torch.float16),
                            "d_fused": d_fused.detach().cpu().to(torch.float16),
                            "d_perp": d_perp.detach().cpu().to(torch.float16),
                            "d_fwd_aux": d_fwd_aux.detach().cpu().to(torch.float16),
                            "m": m.detach().cpu().to(torch.float16),
                            "input_start_path": input_start_path,
                            "input_end_path": input_end_path,
                            "num_frames": int(num_frames),
                            "num_steps": int(num_steps),
                        }
                        torch.save(snapshot, snapshots_dir / f"step_{active_counter:03d}.pt")

                    active_counter += 1
                    x = torch.flip(x, dims=[0])
                else:
                    x_u, x_c = denoiser(x, sigma_hat, cond, uc_loop)
                    denoised = cfg(x_u, x_c, scale=cfg_scale)
                    denoised_hat = dds(denoised, n_inner=dds_cg_inner, latent_target=latent_end)
                    d = (x - x_u) / append_dims(sigma_hat, x.ndim)
                    dt = append_dims(next_sigma, x.ndim)
                    x = denoised_hat + d * dt

                    eps_noise = torch.randn_like(x) * model.sampler.s_noise
                    x = x + eps_noise * append_dims(sigma_hat**2 - next_sigma**2, x.ndim) ** 0.5
                    x = torch.flip(x, dims=[0])

                    x_u, x_c = denoiser(x, sigma_hat, c_end, uc_end)
                    denoised = cfg(x_u, x_c, scale=cfg_scale_flip)
                    denoised_hat = dds(denoised, n_inner=dds_cg_inner, latent_target=latent)
                    d = (x - x_u) / append_dims(sigma_hat, x.ndim)
                    dt = append_dims(next_sigma, x.ndim)
                    x = denoised_hat + d * dt
                    x = torch.flip(x, dims=[0])

    analysis_case_dir = ensure_dir(analysis_root / case_name)
    save_json(meta, analysis_case_dir / "meta.json")
    if save_metrics:
        save_metrics_csv(active_step_records, analysis_case_dir / "metrics.csv")

    if save_video:
        model.en_and_decode_n_samples_a_time = decoding_t
        x_decode = x.to(device=device, dtype=torch.float32)
        model = model.to(torch.float32)
        samples_x = model.decode_first_stage(x_decode)
        samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
        model = model.to(torch.float16)

        samples = embed_watermark(samples)
        if filter_model is not None:
            samples = filter_model(samples)

        vid = (rearrange(samples, "t c h w -> t h w c") * 255).cpu().numpy().astype(np.uint8)
        base_count = len(glob(str(output_case_dir / "*.gif")))
        video_path = output_case_dir / f"{base_count:06d}.gif"

        images = [Image.fromarray(vid[i]) for i in range(vid.shape[0])]
        images[0].save(video_path, save_all=True, append_images=images[1:], duration=125, loop=0)
        print(f"[{case_name}] Saved video to {video_path}")

    if save_metrics:
        print(f"[{case_name}] Saved metrics to {analysis_case_dir / 'metrics.csv'}")
    else:
        print(f"[{case_name}] Metrics saving was skipped.")

    if save_snapshots:
        print(f"[{case_name}] Saved snapshots to {snapshots_dir}")
    else:
        print(f"[{case_name}] Snapshot saving was skipped.")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MR-CATR and save rebuttal diagnostics for visualization.")
    parser.add_argument("--data_root", type=str, default="", help="Folder that contains case subfolders such as train/, bear/, boat/, car-turn/.")
    parser.add_argument("--case_file", type=str, default="", help="Optional JSON file with a list of {'name','start','end'} objects.")
    parser.add_argument("--cases", type=str, default="all", help="Comma-separated case names to run when using --data_root. Use 'all' for the default demo cases.")
    parser.add_argument("--case_name", type=str, default="", help="Name for single-case mode when --input_start_path and --input_end_path are provided.")
    parser.add_argument("--input_start_path", type=str, default="", help="Run a single custom case using this start image path.")
    parser.add_argument("--input_end_path", type=str, default="", help="Run a single custom case using this end image path.")
    parser.add_argument("--output_root", type=str, required=True, help="Folder where final GIFs are stored.")
    parser.add_argument("--analysis_root", type=str, required=True, help="Folder where diagnostics, CSVs, and snapshots are stored.")
    parser.add_argument("--config", type=str, default="scripts/sampling/configs/svd_xt.yaml", help="Path to the sampler config that contains the SVD checkpoint path.")
    parser.add_argument("--version", type=str, default="svd_xt")
    parser.add_argument("--num_frames", type=int, default=25)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--fps_id", type=int, default=10)
    parser.add_argument("--motion_bucket_id", type=int, default=127)
    parser.add_argument("--cond_aug", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--decoding_t", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--cfg_scale_flip", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--active_fraction", type=float, default=0.2, help="Fraction of the denoising steps where MR-CATR projection is active in your current code.")
    parser.add_argument("--snapshot_steps", type=str, default="auto", help="Comma-separated active-step ranks to save. Default: auto -> first/middle/last active steps.")
    parser.add_argument("--dds_cg_inner", type=int, default=5)
    parser.add_argument("--skip_snapshots", action="store_true", help="Do not save full latent snapshots.")
    parser.add_argument("--skip_metrics", action="store_true", help="Do not save per-step CSV metrics.")
    parser.add_argument("--skip_video", action="store_true", help="Skip final video decoding to save time and VRAM.")
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    output_root = ensure_dir(Path(args.output_root))
    analysis_root = ensure_dir(Path(args.analysis_root))

    single_case_mode = bool(args.input_start_path and args.input_end_path)
    if single_case_mode:
        case_name = args.case_name.strip() or "custom_case"
        cases = [
            {
                "name": case_name,
                "start": args.input_start_path,
                "end": args.input_end_path,
            }
        ]
    elif args.case_file:
        cases = load_case_file(Path(args.case_file))
    else:
        if not args.data_root:
            raise ValueError("You must provide either --input_start_path/--input_end_path, or --case_file, or --data_root.")
        cases = choose_cases_from_defaults(Path(args.data_root), parse_csv_list(args.cases))

    snapshot_steps = parse_snapshot_steps(args.snapshot_steps, args.num_steps, args.active_fraction)
    print(f"Snapshot active-step ranks to save: {snapshot_steps}")

    model, filter_model = load_model(
        config=args.config,
        device=args.device,
        num_frames=args.num_frames,
        num_steps=args.num_steps,
        verbose=args.verbose,
        with_filter=not args.skip_video,
    )

    for case in cases:
        sample_case(
            model=model,
            filter_model=filter_model,
            case_name=case["name"],
            input_start_path=case["start"],
            input_end_path=case["end"],
            output_root=output_root,
            analysis_root=analysis_root,
            num_frames=args.num_frames,
            num_steps=args.num_steps,
            version=args.version,
            fps_id=args.fps_id,
            motion_bucket_id=args.motion_bucket_id,
            cond_aug=args.cond_aug,
            seed=args.seed,
            decoding_t=args.decoding_t,
            device=args.device,
            cfg_scale=args.cfg_scale,
            cfg_scale_flip=args.cfg_scale_flip,
            verbose=args.verbose,
            active_fraction=args.active_fraction,
            snapshot_steps=snapshot_steps,
            save_snapshots=not args.skip_snapshots,
            save_metrics=not args.skip_metrics,
            save_video=not args.skip_video,
            dds_cg_inner=args.dds_cg_inner,
        )

    print("All requested cases finished.")
    print(f"Diagnostics root: {analysis_root}")
    print(f"Output root: {output_root}")


if __name__ == "__main__":
    main()
