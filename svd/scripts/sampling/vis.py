import math
import os
import sys
from glob import glob
from contextlib import nullcontext
from typing import Optional

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm

from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config, append_dims


# =========================
# Basic math helpers
# =========================
def _dot(a, b, eps=1e-8):
    return torch.sum(a.float() * b.float())

def _norm(a, eps=1e-8):
    return torch.sqrt(torch.sum(a.float() * a.float()) + eps)

def _normalize(v, eps=1e-8):
    return v / (_norm(v, eps=eps) + eps)

def _cos(a, b, eps=1e-8):
    return _dot(a, b, eps=eps) / (_norm(a, eps=eps) * _norm(b, eps=eps) + eps)


# =========================
# MR-CATR core helpers
# =========================
def motion_axis_from_denoised(denoised, eps=1e-8):
    # denoised: (T,C,H,W)
    res = denoised[1:] - denoised[:-1]
    m = torch.mean(res, dim=0)  # (C,H,W)
    m = m.unsqueeze(0).repeat(denoised.shape[0], 1, 1, 1)  # (T,C,H,W)
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

def conflict_aware_fuse(d_bwd, d_bwd_aligned, sigma, sigma_T, p=2.0, eps=1e-8):
    s = _cos(d_bwd, d_bwd_aligned, eps=eps)
    s = torch.clamp(s, min=0.0)
    rho = rho_schedule(sigma, sigma_T, p=p)
    beta = rho * s
    d = (1 - beta) * d_bwd + beta * d_bwd_aligned
    return d, beta, s, rho


# =========================
# DDS helpers (same logic as your code)
# =========================
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

def DDS(x, n_inner, latent):
    measurement = torch.zeros_like(x)
    measurement[-1, :, :, :] = latent
    A = lambda z: masking(z, -1)
    AT = lambda z: masking(z, -1)
    def Acg(xx):
        return AT(A(xx))
    bcg = AT(measurement)
    return CG(Acg, bcg, x, n_inner=n_inner)


# =========================
# Visualization helpers
# =========================
def _flatten(v):
    return v.detach().float().reshape(-1)

def make_plane_basis(v1, v2, v3=None, eps=1e-8):
    e1 = _flatten(v1)
    e1 = e1 / (e1.norm() + eps)

    cand = _flatten(v2) - torch.dot(_flatten(v2), e1) * e1
    if cand.norm() < eps:
        if v3 is None:
            v3 = torch.randn_like(v1)
        cand = _flatten(v3) - torch.dot(_flatten(v3), e1) * e1
    e2 = cand / (cand.norm() + eps)
    return e1, e2

def plane_coords(v, e1, e2, normalize_vec=True, eps=1e-8):
    vv = _normalize(v, eps=eps) if normalize_vec else v
    vv = _flatten(vv)
    return float(torch.dot(vv, e1)), float(torch.dot(vv, e2))

def choose_rep_frame_from_latent_residual(denoised_start_rev):
    # choose k maximizing latent residual magnitude between k-1 and k
    # denoised_start_rev: (T,C,H,W)
    energy = (denoised_start_rev[1:] - denoised_start_rev[:-1]).abs().mean(dim=(1, 2, 3))
    k = int(torch.argmax(energy).item()) + 1
    k = max(1, min(k, denoised_start_rev.shape[0] - 1))
    return k


@torch.no_grad()
def decode_selected_frames(model, z_cpu, frame_ids, device="cuda"):
    """
    z_cpu: CPU tensor with shape (T,C,H,W)
    frame_ids: list[int]
    model: already moved to `device` and float32 BEFORE calling this function
    """
    frame_ids = sorted(set(int(i) for i in frame_ids))
    z_sub = z_cpu[frame_ids].to(device=device, dtype=torch.float32)
    x = model.decode_first_stage(z_sub)
    x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0).detach().cpu()
    return {fid: x[idx] for idx, fid in enumerate(frame_ids)}

def diff_heatmap(img_a, img_b):
    # img_a, img_b: (3,H,W), output (H,W)
    return (img_b - img_a).abs().mean(dim=0).numpy()

def auto_crop_maps(*maps, pad=24, q=0.98):
    union = np.maximum.reduce(maps)
    thr = np.quantile(union, q)
    ys, xs = np.where(union >= thr)
    if len(xs) == 0:
        h, w = union.shape
        return slice(0, h), slice(0, w)
    y0 = max(0, ys.min() - pad)
    y1 = min(union.shape[0], ys.max() + pad)
    x0 = max(0, xs.min() - pad)
    x1 = min(union.shape[1], xs.max() + pad)
    return slice(y0, y1), slice(x0, x1)


def save_step_visualization(
    model,
    save_png,
    save_pt,
    device,
    step_idx,
    active,
    sigma_val,
    beta,
    rho,
    s_prior,
    s_agree,
    x_base,
    d_fwd_ref,
    d_bwd,
    d_bwd_aligned,
    m,
    denoised_start_rev,
    viz_scale=10.0,
    force_align_to_ref=True,
):
    # -------- save tensors --------
    torch.save(
        {
            "step": step_idx,
            "active": active,
            "sigma": float(sigma_val),
            "beta": float(beta),
            "rho": float(rho),
            "s_prior": float(s_prior),
            "s_agree": float(s_agree),
            "x_base": x_base.detach().cpu().float(),
            "d_fwd_ref": d_fwd_ref.detach().cpu().float(),
            "d_bwd": d_bwd.detach().cpu().float(),
            "d_bwd_aligned": d_bwd_aligned.detach().cpu().float(),
            "m": m.detach().cpu().float(),
            "denoised_start_rev": denoised_start_rev.detach().cpu().float(),
        },
        save_pt,
    )

    # -------- choose representative frame --------
    k = choose_rep_frame_from_latent_residual(denoised_start_rev)

    # -------- aligned-for-visualization only --------
    d_ali_viz = d_bwd_aligned.clone()
    if force_align_to_ref:
        if _dot(d_ali_viz, d_fwd_ref).item() < 0:
            d_ali_viz = -d_ali_viz

    # -------- decode only needed frames --------
    start_rgb = decode_selected_frames(model, denoised_start_rev, [k - 1, k], device=device)
    base_rgb  = decode_selected_frames(model, x_base, [k], device=device)
    raw_rgb   = decode_selected_frames(model, x_base + viz_scale * d_bwd, [k], device=device)
    ali_rgb   = decode_selected_frames(model, x_base + viz_scale * d_ali_viz, [k], device=device)

    res_start = diff_heatmap(start_rgb[k - 1], start_rgb[k])
    res_raw   = diff_heatmap(base_rgb[k], raw_rgb[k])
    res_ali   = diff_heatmap(base_rgb[k], ali_rgb[k])

    ysl, xsl = auto_crop_maps(res_start, res_raw, res_ali, pad=32, q=0.985)
    res_start_c = res_start[ysl, xsl]
    res_raw_c   = res_raw[ysl, xsl]
    res_ali_c   = res_ali[ysl, xsl]

    vmax_start = max(float(res_start_c.max()), 1e-8)
    vmax_raw   = max(float(res_raw_c.max()), 1e-8)
    vmax_ali   = max(float(res_ali_c.max()), 1e-8)

    # -------- plot only heatmaps 2/3/4, no titles --------
    # fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # im1 = axes[0].imshow(res_start_c, cmap="magma", vmin=0, vmax=vmax_start)
    # axes[0].axis("off")

    # im2 = axes[1].imshow(res_raw_c, cmap="magma", vmin=0, vmax=vmax_raw)
    # axes[1].axis("off")

    # im3 = axes[2].imshow(res_ali_c, cmap="magma", vmin=0, vmax=vmax_ali)
    # axes[2].axis("off")

    # fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    # fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    # fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # plt.tight_layout()
    # plt.savefig(save_png, dpi=240, bbox_inches="tight")
    # plt.close(fig)
    

    fig = plt.figure(figsize=(14, 4))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.06, 1, 1, 1], wspace=0.08)

    cax = fig.add_subplot(gs[0, 0])
    ax0 = fig.add_subplot(gs[0, 1])
    ax1 = fig.add_subplot(gs[0, 2])
    ax2 = fig.add_subplot(gs[0, 3])

    im1 = ax0.imshow(res_start_c, cmap="magma", vmin=0, vmax=vmax_start)
    ax0.axis("off")

    im2 = ax1.imshow(res_raw_c, cmap="magma", vmin=0, vmax=vmax_raw)
    ax1.axis("off")

    im3 = ax2.imshow(res_ali_c, cmap="magma", vmin=0, vmax=vmax_ali)
    ax2.axis("off")

    cb = fig.colorbar(im1, cax=cax)

    # 把刻度放到左边，避免和第一张图挤在一起
    cb.ax.yaxis.set_ticks_position("left")
    cb.ax.yaxis.set_label_position("left")

    # 给左侧刻度数字留空间
    fig.subplots_adjust(left=0.10, right=0.98)

    plt.savefig(save_png, dpi=240, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
        
# =========================
# Conditioning / model load
# =========================
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

def load_model(config, device, num_frames, num_steps, verbose=False):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.verbose = verbose
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = num_frames

    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    model = model.to(torch.float16)
    filter_model = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter_model


# =========================
# Main sampling with per-step visualization
# =========================
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
    save_step_viz: bool = True,
):
    """
    Sampling + per-step visualization for Ours+ViBiD.
    IMPORTANT:
      - During sampling: only cache step tensors on CPU.
      - After sampling: exit autocast, move model to float32, then render all step visualizations.
    """

    if version == "svd_xt":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 25)
        model_config = "scripts/sampling/configs/svd_xt.yaml"
    else:
        raise ValueError(f"Version {version} does not exist.")

    model, filter_model = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
        verbose,
    )
    torch.manual_seed(seed)

    # ---------- load start frame ----------
    with Image.open(input_start_path) as image:
        input_image = image.convert("RGB")
        input_image = input_image.resize((1024, 576))
        w, h = input_image.size

        if h % 64 != 0 or w % 64 != 0:
            width, height = map(lambda x: x - x % 64, (w, h))
            input_image = input_image.resize((width, height))
            print(
                f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. "
                f"We are resizing to {height}x{width}!"
            )

    image = ToTensor()(input_image)
    image = image * 2.0 - 1.0
    image = image.unsqueeze(0).to(device).to(torch.float16)
    latent = model.encode_first_stage(image)

    # ---------- load end frame ----------
    input_image_end = Image.open(input_end_path).convert("RGB").resize((1024, 576))
    image_end = ToTensor()(input_image_end)
    image_end = image_end * 2.0 - 1.0
    image_end = image_end.unsqueeze(0).to(device).to(torch.float16)
    latent_end = model.encode_first_stage(image_end)

    H, W = image.shape[2:]
    assert image.shape[1] == 3
    F = 8
    C = 4
    shape = (num_frames, C, H // F, W // F)

    if motion_bucket_id > 255:
        print("WARNING: High motion bucket! This may lead to suboptimal performance.")
    if fps_id < 5:
        print("WARNING: Small fps value! This may lead to suboptimal performance.")
    if fps_id > 30:
        print("WARNING: Large fps value! This may lead to suboptimal performance.")

    # ---------- start condition ----------
    value_dict = {}
    value_dict["cond_frames_without_noise"] = image
    value_dict["motion_bucket_id"] = motion_bucket_id
    value_dict["fps_id"] = fps_id
    value_dict["cond_aug"] = cond_aug
    value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)

    # ---------- end condition ----------
    value_dict_end = {}
    value_dict_end["cond_frames_without_noise"] = image_end
    value_dict_end["motion_bucket_id"] = motion_bucket_id
    value_dict_end["fps_id"] = fps_id
    value_dict_end["cond_aug"] = cond_aug
    value_dict_end["cond_frames"] = image_end + cond_aug * torch.randn_like(image_end)

    # ---------- output dirs ----------
    if output_folder is None:
        output_folder = "./outputs"
    os.makedirs(output_folder, exist_ok=True)

    step_viz_dir = os.path.join(output_folder, "step_viz")
    step_png_dir = os.path.join(step_viz_dir, "png")
    step_pt_dir = os.path.join(step_viz_dir, "pt")
    if save_step_viz:
        os.makedirs(step_png_dir, exist_ok=True)
        os.makedirs(step_pt_dir, exist_ok=True)

    step_records = []

    with torch.no_grad():
        # =========================
        # Sampling phase (autocast on)
        # =========================
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if "cuda" in device
            else nullcontext()
        )

        with autocast_ctx:
            # ---------- conditioning ----------
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
                force_uc_zero_embeddings=[
                    "cond_frames",
                    "cond_frames_without_noise",
                ],
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
                force_uc_zero_embeddings=[
                    "cond_frames",
                    "cond_frames_without_noise",
                ],
            )

            for k in ["crossattn", "concat"]:
                uc_end[k] = repeat(uc_end[k], "b ... -> b t ...", t=num_frames)
                uc_end[k] = rearrange(uc_end[k], "b t ... -> (b t) ...", t=num_frames)
                c_end[k] = repeat(c_end[k], "b ... -> b t ...", t=num_frames)
                c_end[k] = rearrange(c_end[k], "b t ... -> (b t) ...", t=num_frames)

            randn = torch.randn(shape, device=device)

            additional_model_inputs = {}
            additional_model_inputs["image_only_indicator"] = torch.zeros(
                2, num_frames
            ).to(device)
            additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

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
                sigma_shape = denoiser_sigma.shape

                denoiser_sigma_exp = append_dims(denoiser_sigma, x.ndim)
                c_skip = 1.0 / (denoiser_sigma_exp**2 + 1.0)
                c_out_scale = -denoiser_sigma_exp / (denoiser_sigma_exp**2 + 1.0) ** 0.5
                c_in = 1.0 / (denoiser_sigma_exp**2 + 1.0) ** 0.5
                c_noise = 0.25 * denoiser_sigma_exp.log()
                c_noise = c_noise.reshape(sigma_shape)

                denoised = (
                    model.model(
                        denoiser_input * c_in,
                        c_noise,
                        c_out,
                        **additional_model_inputs,
                    )
                    * c_out_scale
                    + denoiser_input * c_skip
                )

                x_u, x_c = denoised.chunk(2)
                return x_u, x_c

            def CFG(x_u, x_c, scale):
                x_u = rearrange(x_u, "(b t) ... -> b t ...", t=num_frames)
                x_c = rearrange(x_c, "(b t) ... -> b t ...", t=num_frames)
                scale_vec = torch.linspace(scale, scale, steps=num_frames).unsqueeze(0)
                scale_vec = repeat(scale_vec, "1 t -> b t", b=x_u.shape[0])
                scale_vec = append_dims(scale_vec, x_u.ndim).to(x_u.device)
                denoised = rearrange(
                    x_u + scale_vec * (x_c - x_u),
                    "b t ... -> (b t) ...",
                )
                return denoised

            x, s_in, sigmas, num_sigmas, cond, uc = model.sampler.prepare_sampling_loop(
                randn, c, uc, num_steps
            )

            for i in tqdm(model.sampler.get_sigma_gen(num_sigmas), total=num_sigmas - 1):
                gamma = (
                    min(model.sampler.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                    if model.sampler.s_tmin <= sigmas[i] <= model.sampler.s_tmax
                    else 0.0
                )

                sigma = s_in * sigmas[i]
                next_sigma = s_in * sigmas[i + 1]
                sigma_hat = sigma * (gamma + 1.0)

                if gamma > 0:
                    eps = torch.randn_like(x) * model.sampler.s_noise
                    x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

                # ---------- common forward half-step ----------
                x_in = x.clone()

                x_u, x_c = denoiser(x, sigma_hat, cond, uc)
                denoised_start = CFG(x_u, x_c, scale=cfg_scale)
                denoised_start_for_axis = denoised_start.detach()

                denoised_hat_start = DDS(denoised_start, n_inner=5, latent=latent_end)
                d_solver_start = (x - x_u) / append_dims(sigma_hat, x.ndim)
                dt = append_dims(next_sigma, x.ndim)
                x_fwd = denoised_hat_start + d_solver_start * dt

                # reversed-coordinate forward reference, for visualization only
                d_fwd_ref = torch.flip(x_fwd - x_in, dims=[0])

                # ---------- re-noise + reverse ----------
                eps = torch.randn_like(x_fwd) * model.sampler.s_noise
                x_noised = x_fwd + eps * append_dims(sigma_hat**2 - next_sigma**2, x.ndim) ** 0.5
                x_base = torch.flip(x_noised, dims=[0])

                # ---------- end-conditioned backward half-step ----------
                x_u_end, x_c_end = denoiser(x_base, sigma_hat, c_end, uc_end)
                denoised_end = CFG(x_u_end, x_c_end, scale=cfg_scale_flip)

                denoised_start_rev = torch.flip(denoised_start_for_axis, dims=[0])
                m_start = motion_axis_from_denoised(denoised_start_rev)
                m_end = motion_axis_from_denoised(denoised_end.detach())
                m, s_prior = consensus_axis(m_start, m_end, kappa=0.2)

                denoised_hat_end = DDS(denoised_end, n_inner=5, latent=latent)
                d_solver_end = (x_base - x_u_end) / append_dims(sigma_hat, x.ndim)
                x_raw_next = denoised_hat_end + d_solver_end * dt
                d_bwd = x_raw_next - x_base

                d_bwd_aligned = project_update(d_bwd, m)
                d_fused, beta, s_agree, rho = conflict_aware_fuse(
                    d_bwd,
                    d_bwd_aligned,
                    sigmas[i],
                    sigmas[0],
                    p=2.0,
                )

                active = bool(i < 0.2 * num_steps)

                if save_step_viz:
                    record = {
                        "step": int(i),
                        "active": active,
                        "sigma": float(sigmas[i].item()),
                        "beta": float(beta.item()) if torch.is_tensor(beta) else float(beta),
                        "rho": float(rho.item()) if torch.is_tensor(rho) else float(rho),
                        "s_prior": float(s_prior.item()) if torch.is_tensor(s_prior) else float(s_prior),
                        "s_agree": float(s_agree.item()) if torch.is_tensor(s_agree) else float(s_agree),
                        "x_base": x_base.detach().cpu().float(),
                        "d_fwd_ref": d_fwd_ref.detach().cpu().float(),
                        "d_bwd": d_bwd.detach().cpu().float(),
                        "d_bwd_aligned": d_bwd_aligned.detach().cpu().float(),
                        "m": m.detach().cpu().float(),
                        "denoised_start_rev": denoised_start_rev.detach().cpu().float(),
                    }
                    step_records.append(record)
                    torch.save(record, os.path.join(step_pt_dir, f"step_{int(i):03d}.pt"))

                # ---------- update ----------
                if active:
                    x_rev_next = x_base + d_fused
                else:
                    # fallback to original ViBiD-like backward half-step
                    x_rev_next = x_raw_next

                x = torch.flip(x_rev_next, dims=[0])

            samples_z = x

        # =========================
        # Rendering phase (autocast off, model float32)
        # =========================
        model.en_and_decode_n_samples_a_time = decoding_t
        model = model.to(device).to(torch.float32)

        # final decode
        samples_x = model.decode_first_stage(samples_z.to(device=device, dtype=torch.float32))
        samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

        base_count = len(glob(os.path.join(output_folder, "*.gif")))
        samples = embed_watermark(samples)
        samples = filter_model(samples)

        vid = (
            (rearrange(samples, "t c h w -> t h w c") * 255)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        video_path = os.path.join(output_folder, f"{base_count:06d}.gif")
        images = [Image.fromarray(vid[j]) for j in range(vid.shape[0])]
        images[0].save(
            video_path,
            save_all=True,
            append_images=images[1:],
            duration=125,
            loop=0,
        )

        # render step visualizations after sampling
        if save_step_viz:
            print(f"[Info] Rendering {len(step_records)} step visualizations...")
            for rec in tqdm(step_records, desc="render step viz"):
                save_png = os.path.join(step_png_dir, f"step_{rec['step']:03d}.png")
                save_pt = os.path.join(step_pt_dir, f"step_{rec['step']:03d}.pt")
                save_step_visualization(
                    model=model,
                    save_png=save_png,
                    save_pt=save_pt,
                    device=device,
                    step_idx=rec["step"],
                    active=rec["active"],
                    sigma_val=rec["sigma"],
                    beta=rec["beta"],
                    rho=rec["rho"],
                    s_prior=rec["s_prior"],
                    s_agree=rec["s_agree"],
                    x_base=rec["x_base"],
                    d_fwd_ref=rec["d_fwd_ref"],
                    d_bwd=rec["d_bwd"],
                    d_bwd_aligned=rec["d_bwd_aligned"],
                    m=rec["m"],
                    denoised_start_rev=rec["denoised_start_rev"],
                )


if __name__ == "__main__":
    DATA_PATH = '/home/zhenbang/code/data' 
    base_output_folder = '/home/zhenbang/code/data_vis'

    cases = [
        {
            "name": "train",
            "start": f"{DATA_PATH}/train/00000.jpg",
            "end": f"{DATA_PATH}/train/00024.jpg",
        },
        {
            "name": "bear",
            "start": f"{DATA_PATH}/bear/00000.jpg",
            "end": f"{DATA_PATH}/bear/00024.jpg",
        },
        {
            "name": "boat",
            "start": f"{DATA_PATH}/boat/00000.jpg",
            "end": f"{DATA_PATH}/boat/00024.jpg",
        },
        {
            "name": "car-turn",
            "start": f"{DATA_PATH}/car-turn/00000.jpg",
            "end": f"{DATA_PATH}/car-turn/00024.jpg",
        },
    ]

    for case in cases:
        sample(
            input_start_path=case["start"],
            input_end_path=case["end"],
            fps_id=10,
            output_folder=f"{base_output_folder}/{case['name']}",
            save_step_viz=True,
        )