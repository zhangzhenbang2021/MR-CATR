import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from sgm.util import append_dims


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def _dot(a, b, eps: float = 1e-8):
    return torch.sum(a.float() * b.float())


def _norm(a, eps: float = 1e-8):
    return torch.sqrt(torch.sum(a.float() * a.float()) + eps)


def _normalize(v, eps: float = 1e-8):
    return v / (_norm(v, eps=eps) + eps)


def _cos(a, b, eps: float = 1e-8):
    return _dot(a, b, eps=eps) / (_norm(a, eps=eps) * _norm(b, eps=eps) + eps)


def device_type_from_device(device: str) -> str:
    return "cuda" if str(device).startswith("cuda") else "cpu"


def local_cosine_map(u: torch.Tensor, v: torch.Tensor, kt: int = 3, kh: int = 5, kw: int = 5, eps: float = 1e-8) -> torch.Tensor:
    prod = torch.sum(u.float() * v.float(), dim=1)
    uu = torch.sum(u.float() * u.float(), dim=1)
    vv = torch.sum(v.float() * v.float(), dim=1)

    def _pool(x):
        x = x.unsqueeze(0).unsqueeze(0)
        y = F.avg_pool3d(x, kernel_size=(kt, kh, kw), stride=1, padding=(kt // 2, kh // 2, kw // 2))
        return y[0, 0]

    prod_p = _pool(prod)
    uu_p = _pool(uu)
    vv_p = _pool(vv)
    return prod_p / torch.sqrt(uu_p * vv_p + eps)


# -----------------------------------------------------------------------------
# Lazy model load
# -----------------------------------------------------------------------------

def lazy_load_model(*args, **kwargs):
    from scripts.sampling.mrcatr_truefwd_diagnostics import load_model
    return load_model(*args, **kwargs)


# -----------------------------------------------------------------------------
# Basic IO helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_csv_list(text: Optional[str]) -> List[str]:
    if text is None:
        return []
    text = text.strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_frame_indices(text: str, num_frames: int) -> List[int]:
    text = text.strip().lower()
    if text == "auto":
        if num_frames <= 3:
            return list(range(num_frames))
        return sorted(set([0, num_frames // 4, num_frames // 2, (3 * num_frames) // 4, num_frames - 1]))
    indices = [int(item) for item in parse_csv_list(text)]
    if not indices:
        raise ValueError("frame_indices resolved to an empty list.")
    return indices


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metrics_csv(path: Path) -> List[Dict]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            converted = {}
            for key, value in row.items():
                if key == "case_name":
                    converted[key] = value
                elif key in {"loop_step", "active_step_rank"}:
                    converted[key] = int(value)
                else:
                    converted[key] = float(value)
            rows.append(converted)
        return rows


def save_text(text: str, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_csv(rows: List[Dict], path: Path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------------------------------------------------------
# Discovery helpers
# -----------------------------------------------------------------------------

def find_case_dirs(analysis_root: Path, case_filter: Sequence[str]) -> List[Path]:
    case_filter_set = set(case_filter)
    case_dirs = []
    for child in sorted(analysis_root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "metrics.csv").exists():
            continue
        if case_filter_set and child.name not in case_filter_set:
            continue
        case_dirs.append(child)
    return case_dirs


def collect_all_metrics(case_dirs: Sequence[Path]) -> List[Dict]:
    rows = []
    for case_dir in case_dirs:
        rows.extend(load_metrics_csv(case_dir / "metrics.csv"))
    return rows


def list_snapshot_paths(case_dir: Path) -> List[Path]:
    snap_dir = case_dir / "snapshots"
    if not snap_dir.exists():
        return []
    return sorted(snap_dir.glob("step_*.pt"))


def choose_snapshot_path(case_dirs: Sequence[Path], ranking_rows: Sequence[Dict], explicit_case: str, explicit_step: Optional[int]) -> Tuple[str, Path]:
    if explicit_case and explicit_step is not None:
        path = Path(case_dirs[[c.name for c in case_dirs].index(explicit_case)] / "snapshots" / f"step_{explicit_step:03d}.pt")
        if not path.exists():
            raise FileNotFoundError(f"Snapshot not found: {path}")
        return explicit_case, path

    for row in ranking_rows:
        case_name = row["case_name"]
        step = int(row["active_step_rank"])
        for case_dir in case_dirs:
            if case_dir.name != case_name:
                continue
            path = case_dir / "snapshots" / f"step_{step:03d}.pt"
            if path.exists():
                return case_name, path
    raise ValueError("No ranked snapshot with a saved .pt file was found.")


# -----------------------------------------------------------------------------
# Summary and ranking
# -----------------------------------------------------------------------------

def aggregate_by_active_step(rows: Sequence[Dict]) -> Dict[int, Dict[str, np.ndarray]]:
    grouped: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        k = int(row["active_step_rank"])
        for metric in [
            "motion_mean_cos_raw",
            "motion_mean_cos_disp",
            "motion_mean_cos_res",
            "motion_gain_disp_vs_raw",
            "motion_gain_res_vs_raw",
            "motion_gain_res_vs_disp",
            "beta",
            "s_prior",
        ]:
            grouped[k][metric].append(float(row[metric]))
    result = {}
    for step, metrics in grouped.items():
        result[step] = {name: np.asarray(values, dtype=np.float64) for name, values in metrics.items()}
    return result


def build_rankings(rows: Sequence[Dict]) -> Tuple[List[Dict], List[Dict], str]:
    if not rows:
        return [], [], "No metrics rows were found.\n"

    case_groups: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        case_groups[row["case_name"]].append(row)

    case_ranking_rows: List[Dict] = []
    for case_name, case_rows in sorted(case_groups.items()):
        case_ranking_rows.append(
            {
                "case_name": case_name,
                "n_steps": len(case_rows),
                "mean_motion_cos_raw": float(np.mean([r["motion_mean_cos_raw"] for r in case_rows])),
                "mean_motion_cos_disp": float(np.mean([r["motion_mean_cos_disp"] for r in case_rows])),
                "mean_motion_cos_res": float(np.mean([r["motion_mean_cos_res"] for r in case_rows])),
                "mean_gain_disp_vs_raw": float(np.mean([r["motion_gain_disp_vs_raw"] for r in case_rows])),
                "mean_gain_res_vs_raw": float(np.mean([r["motion_gain_res_vs_raw"] for r in case_rows])),
                "mean_gain_res_vs_disp": float(np.mean([r["motion_gain_res_vs_disp"] for r in case_rows])),
                "mean_beta": float(np.mean([r["beta"] for r in case_rows])),
                "mean_s_prior": float(np.mean([r["s_prior"] for r in case_rows])),
            }
        )
    case_ranking_rows.sort(key=lambda r: (r["mean_gain_res_vs_disp"], r["mean_gain_res_vs_raw"], r["mean_motion_cos_res"]), reverse=True)

    snapshot_ranking_rows: List[Dict] = []
    for row in rows:
        snapshot_ranking_rows.append(
            {
                "case_name": row["case_name"],
                "active_step_rank": int(row["active_step_rank"]),
                "loop_step": int(row["loop_step"]),
                "motion_mean_cos_raw": float(row["motion_mean_cos_raw"]),
                "motion_mean_cos_disp": float(row["motion_mean_cos_disp"]),
                "motion_mean_cos_res": float(row["motion_mean_cos_res"]),
                "motion_gain_disp_vs_raw": float(row["motion_gain_disp_vs_raw"]),
                "motion_gain_res_vs_raw": float(row["motion_gain_res_vs_raw"]),
                "motion_gain_res_vs_disp": float(row["motion_gain_res_vs_disp"]),
                "beta": float(row["beta"]),
                "s_prior": float(row["s_prior"]),
                "score": float(row["motion_gain_res_vs_disp"] + 0.5 * row["motion_gain_res_vs_raw"]),
            }
        )
    snapshot_ranking_rows.sort(key=lambda r: (r["score"], r["motion_mean_cos_res"], r["s_prior"]), reverse=True)

    lines = []
    lines.append("MR-CATR true-forward rebuttal summary\n")
    lines.append(f"Active-step records analyzed: {len(rows)}")
    lines.append(f"Mean motion-region local cosine, raw: {np.mean([r['motion_mean_cos_raw'] for r in rows]):.4f}")
    lines.append(f"Mean motion-region local cosine, simple displacement: {np.mean([r['motion_mean_cos_disp'] for r in rows]):.4f}")
    lines.append(f"Mean motion-region local cosine, residual axis: {np.mean([r['motion_mean_cos_res'] for r in rows]):.4f}")
    lines.append(f"Mean motion-region gain, simple vs raw: {np.mean([r['motion_gain_disp_vs_raw'] for r in rows]):.4f}")
    lines.append(f"Mean motion-region gain, residual vs raw: {np.mean([r['motion_gain_res_vs_raw'] for r in rows]):.4f}")
    lines.append(f"Mean motion-region gain, residual vs simple: {np.mean([r['motion_gain_res_vs_disp'] for r in rows]):.4f}")
    lines.append("")
    lines.append("Top representative snapshots:")
    for row in snapshot_ranking_rows[:10]:
        lines.append(
            f"  - {row['case_name']} step={row['active_step_rank']} loop={row['loop_step']} score={row['score']:.4f} "
            f"res={row['motion_mean_cos_res']:.4f} disp={row['motion_mean_cos_disp']:.4f} raw={row['motion_mean_cos_raw']:.4f} "
            f"gain_res_disp={row['motion_gain_res_vs_disp']:.4f}"
        )
    return case_ranking_rows, snapshot_ranking_rows, "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Global figure
# -----------------------------------------------------------------------------

def plot_global_true_forward_alignment(rows: Sequence[Dict], out_path: Path):
    grouped = aggregate_by_active_step(rows)
    steps = sorted(grouped.keys())
    if not steps:
        raise ValueError("No rows found for global plot.")

    raw_vals = [row["motion_mean_cos_raw"] for row in rows]
    disp_vals = [row["motion_mean_cos_disp"] for row in rows]
    res_vals = [row["motion_mean_cos_res"] for row in rows]
    gain_disp = [row["motion_gain_disp_vs_raw"] for row in rows]
    gain_res = [row["motion_gain_res_vs_raw"] for row in rows]
    gain_res_disp = [row["motion_gain_res_vs_disp"] for row in rows]

    plt.figure(figsize=(12, 9))

    ax1 = plt.subplot(2, 2, 1)
    bp1 = ax1.boxplot([raw_vals, disp_vals, res_vals], patch_artist=True, labels=["Raw", "Simple disp", "Residual axis"])
    ax1.set_title("Motion-region local cosine to true forward update")
    ax1.set_ylabel("Mean local cosine")
    ax1.grid(True, alpha=0.25)

    ax2 = plt.subplot(2, 2, 2)
    bp2 = ax2.boxplot([gain_disp, gain_res, gain_res_disp], patch_artist=True, labels=["Simple-raw", "Residual-raw", "Residual-simple"])
    ax2.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax2.set_title("Per-step motion-region cosine gain")
    ax2.set_ylabel("Gain in mean local cosine")
    ax2.grid(True, alpha=0.25)

    def mean_std(metric):
        mean = np.asarray([grouped[s][metric].mean() for s in steps])
        std = np.asarray([grouped[s][metric].std() for s in steps])
        return mean, std

    raw_mean, raw_std = mean_std("motion_mean_cos_raw")
    disp_mean, disp_std = mean_std("motion_mean_cos_disp")
    res_mean, res_std = mean_std("motion_mean_cos_res")

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(steps, raw_mean, marker="o", label="Raw backward")
    ax3.plot(steps, disp_mean, marker="o", label="Simple displacement projection")
    ax3.plot(steps, res_mean, marker="o", label="Residual-axis projection")
    ax3.fill_between(steps, raw_mean - raw_std, raw_mean + raw_std, alpha=0.15)
    ax3.fill_between(steps, disp_mean - disp_std, disp_mean + disp_std, alpha=0.15)
    ax3.fill_between(steps, res_mean - res_std, res_mean + res_std, alpha=0.15)
    ax3.set_title("Step-wise mean local cosine to true forward update")
    ax3.set_xlabel("Active-step rank")
    ax3.set_ylabel("Mean local cosine")
    ax3.grid(True, alpha=0.25)
    ax3.legend(fontsize=9)

    gr_mean, gr_std = mean_std("motion_gain_res_vs_raw")
    gd_mean, gd_std = mean_std("motion_gain_disp_vs_raw")
    gsd_mean, gsd_std = mean_std("motion_gain_res_vs_disp")

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(steps, gd_mean, marker="o", label="Simple - raw")
    ax4.plot(steps, gr_mean, marker="o", label="Residual - raw")
    ax4.plot(steps, gsd_mean, marker="o", label="Residual - simple")
    ax4.fill_between(steps, gd_mean - gd_std, gd_mean + gd_std, alpha=0.15)
    ax4.fill_between(steps, gr_mean - gr_std, gr_mean + gr_std, alpha=0.15)
    ax4.fill_between(steps, gsd_mean - gsd_std, gsd_mean + gsd_std, alpha=0.15)
    ax4.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax4.set_title("Step-wise alignment gain")
    ax4.set_xlabel("Active-step rank")
    ax4.set_ylabel("Gain in mean local cosine")
    ax4.grid(True, alpha=0.25)
    ax4.legend(fontsize=9)

    plt.suptitle("True-forward local alignment: raw backward vs simple-vector vs residual-based projection")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Conditioning / denoiser helpers for x0 look-ahead
# -----------------------------------------------------------------------------

def load_forward_conditioning(case_dir: Path):
    payload = torch.load(case_dir / "forward_conditioning.pt", map_location="cpu")
    return payload


def to_device_cond(d: Dict, device: str) -> Dict:
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            out[k] = v.to(device=device, dtype=torch.float16)
        else:
            out[k] = v
    return out


def run_forward_denoiser_cfg(model, x_start_order: torch.Tensor, sigma_scalar: float, cond_payload: Dict, device: str) -> torch.Tensor:
    c = to_device_cond(cond_payload["cond_c"], device)
    uc = to_device_cond(cond_payload["cond_uc"], device)
    num_frames = int(cond_payload["num_frames"])
    additional_model_inputs = {
        "image_only_indicator": torch.zeros(2, num_frames, device=device),
        "num_video_frames": num_frames,
    }

    x = x_start_order.to(device=device, dtype=torch.float16)
    sigma = torch.full((x.shape[0],), float(sigma_scalar), device=device, dtype=torch.float16)

    with torch.no_grad():
        if device_type_from_device(device) == "cuda":
            autocast_ctx = torch.autocast(device_type="cuda")
        else:
            from contextlib import nullcontext
            autocast_ctx = nullcontext()
        with autocast_ctx:
            c_out = {}
            for k in c:
                if k in ["vector", "crossattn", "concat"]:
                    c_out[k] = torch.cat((uc[k], c[k]), 0)
                else:
                    c_out[k] = c[k]
            denoiser_input = torch.cat([x] * 2)
            denoiser_sigma = torch.cat([sigma] * 2)
            sigma_shape = denoiser_sigma.shape
            denoiser_sigma_appended = append_dims(denoiser_sigma, x.ndim)
            c_skip = 1.0 / (denoiser_sigma_appended**2 + 1.0)
            c_out_coeff = -denoiser_sigma_appended / (denoiser_sigma_appended**2 + 1.0) ** 0.5
            c_in = 1.0 / (denoiser_sigma_appended**2 + 1.0) ** 0.5
            c_noise = 0.25 * denoiser_sigma.log().reshape(sigma_shape)
            denoised = model.model(denoiser_input * c_in, c_noise, c_out, **additional_model_inputs) * c_out_coeff + denoiser_input * c_skip
            x_u, x_c = denoised.chunk(2)
            x_u = x_u.reshape(1, num_frames, *x_u.shape[1:])
            x_c = x_c.reshape(1, num_frames, *x_c.shape[1:])
            scale = torch.linspace(float(cond_payload["cfg_scale"]), float(cond_payload["cfg_scale"]), steps=num_frames, device=device).reshape(1, num_frames)
            scale = append_dims(scale, x_u.ndim)
            denoised_cfg = (x_u + scale * (x_c - x_u)).reshape(num_frames, *x.shape[1:])
    return denoised_cfg.detach()


def decode_latents(model, latents: torch.Tensor, device: str, decoding_t: int) -> torch.Tensor:
    model.en_and_decode_n_samples_a_time = decoding_t
    model = model.to(torch.float32)
    latents = latents.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        samples_x = model.decode_first_stage(latents)
    samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).detach().cpu()
    model = model.to(torch.float16)
    return samples


# -----------------------------------------------------------------------------
# Representative figures
# -----------------------------------------------------------------------------

def tensor_to_coords(v: torch.Tensor, e1: torch.Tensor, e2: torch.Tensor, denom: float) -> Tuple[float, float]:
    x = float(_dot(v, e1).detach().cpu().item()) / denom
    y = float(_dot(v, e2).detach().cpu().item()) / denom
    return x, y


def choose_second_basis(primary: torch.Tensor, e1: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    perp = primary - _dot(primary, e1) * e1
    if float(_norm(perp).detach().cpu().item()) > 1e-8:
        return _normalize(perp)
    alt = fallback - _dot(fallback, e1) * e1
    if float(_norm(alt).detach().cpu().item()) > 1e-8:
        return _normalize(alt)
    flat = torch.zeros_like(primary)
    flat.view(-1)[0] = 1.0
    alt = flat - _dot(flat, e1) * e1
    return _normalize(alt)


def plot_dual_plane_alignment(snapshot_path: Path, out_path: Path):
    snap = torch.load(snapshot_path, map_location="cpu")
    d_raw = snap["d_bwd"].float()
    d_disp = snap["d_dispproj"].float()
    d_res = snap["d_resproj"].float()
    d_fwd = snap["d_fwd_ref_bwd"].float()
    m_res = snap["m_res"].float()

    denom = float(_norm(d_raw).detach().cpu().item()) + 1e-8

    plt.figure(figsize=(12, 5.8))

    # Left panel: decomposition wrt residual-based motion axis.
    ax1 = plt.subplot(1, 2, 1)
    e1_left = _normalize(m_res)
    e2_left = choose_second_basis(d_raw, e1_left, d_disp)
    raw_xy = tensor_to_coords(d_raw, e1_left, e2_left, denom)
    disp_xy = tensor_to_coords(d_disp, e1_left, e2_left, denom)
    res_xy = tensor_to_coords(d_res, e1_left, e2_left, denom)
    for xy, label, color in [(raw_xy, "Raw backward", "tab:blue"), (disp_xy, "Simple-vector proj", "tab:orange"), (res_xy, "Residual-axis proj", "tab:green")]:
        ax1.annotate("", xy=xy, xytext=(0, 0), arrowprops=dict(arrowstyle="->", linewidth=2.2, color=color))
        ax1.scatter([xy[0]], [xy[1]], s=70, color=color, label=label)
    ax1.axhline(0.0, linewidth=1.0, alpha=0.5)
    ax1.axvline(0.0, linewidth=1.0, alpha=0.4)
    ax1.set_xlabel(r"Coordinate along residual-based motion axis $m_{res}$")
    ax1.set_ylabel(r"Coordinate along orthogonal conflict direction")
    ax1.set_title("Projection geometry in the real latent space")
    ax1.grid(True, alpha=0.25)
    ax1.set_aspect("equal", adjustable="box")
    ax1.legend(fontsize=9, loc="lower right")

    # Right panel: alignment wrt true forward update.
    ax2 = plt.subplot(1, 2, 2)
    e1_right = _normalize(d_fwd)
    e2_right = choose_second_basis(d_raw, e1_right, d_res)
    raw_xy = tensor_to_coords(d_raw, e1_right, e2_right, denom)
    disp_xy = tensor_to_coords(d_disp, e1_right, e2_right, denom)
    res_xy = tensor_to_coords(d_res, e1_right, e2_right, denom)
    fwd_xy = tensor_to_coords(d_fwd, e1_right, e2_right, denom)
    for xy, label, color in [(raw_xy, "Raw backward", "tab:blue"), (disp_xy, "Simple-vector proj", "tab:orange"), (res_xy, "Residual-axis proj", "tab:green"), (fwd_xy, "True forward", "tab:red")]:
        ax2.annotate("", xy=xy, xytext=(0, 0), arrowprops=dict(arrowstyle="->", linewidth=2.2, color=color))
        ax2.scatter([xy[0]], [xy[1]], s=70, color=color, label=label)
    ax2.axhline(0.0, linewidth=1.0, alpha=0.5)
    ax2.axvline(0.0, linewidth=1.0, alpha=0.4)
    ax2.set_xlabel(r"Coordinate along true forward update $d_{fwd}$")
    ax2.set_ylabel(r"Coordinate orthogonal to $d_{fwd}$")
    ax2.set_title("Which candidate is actually closer to the forward update?")
    ax2.grid(True, alpha=0.25)
    ax2.set_aspect("equal", adjustable="box")
    ax2.legend(fontsize=9, loc="lower right")

    info = (
        f"case={snap['case_name']}  active={snap['active_step_rank']}  loop={snap['loop_step']}\n"
        f"global cos to true forward: raw={float(_cos(d_raw, d_fwd)):.3f}, simple={float(_cos(d_disp, d_fwd)):.3f}, residual={float(_cos(d_res, d_fwd)):.3f}\n"
        f"motion-region local cos: raw={float(snap['motion_mean_cos_raw']):.3f}, simple={float(snap['motion_mean_cos_disp']):.3f}, residual={float(snap['motion_mean_cos_res']):.3f}"
    )
    plt.suptitle(info, fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_x0_lookahead(snapshot_path: Path, case_dir: Path, out_path: Path, model, device: str, decoding_t: int, frame_indices: Sequence[int]):
    snap = torch.load(snapshot_path, map_location="cpu")
    cond_payload = load_forward_conditioning(case_dir)

    sigma_next = float(snap["next_sigma"])
    x_fwd_next_start = snap["x_fwd_next_start"].float()
    x_raw_next_start = torch.flip(snap["x_raw_next_bwd"].float(), dims=[0])
    x_disp_next_start = torch.flip(snap["x_disp_next_bwd"].float(), dims=[0])
    x_res_next_start = torch.flip(snap["x_res_next_bwd"].float(), dims=[0])

    x0_ref = run_forward_denoiser_cfg(model, x_fwd_next_start, sigma_next, cond_payload, device=device)
    x0_raw = run_forward_denoiser_cfg(model, x_raw_next_start, sigma_next, cond_payload, device=device)
    x0_disp = run_forward_denoiser_cfg(model, x_disp_next_start, sigma_next, cond_payload, device=device)
    x0_res = run_forward_denoiser_cfg(model, x_res_next_start, sigma_next, cond_payload, device=device)

    ref_video = decode_latents(model, x0_ref, device=device, decoding_t=decoding_t)
    raw_video = decode_latents(model, x0_raw, device=device, decoding_t=decoding_t)
    disp_video = decode_latents(model, x0_disp, device=device, decoding_t=decoding_t)
    res_video = decode_latents(model, x0_res, device=device, decoding_t=decoding_t)

    diff_raw = torch.mean(torch.abs(raw_video - ref_video), dim=1)
    diff_disp = torch.mean(torch.abs(disp_video - ref_video), dim=1)
    diff_res = torch.mean(torch.abs(res_video - ref_video), dim=1)
    diff_max = max(float(diff_raw.max()), float(diff_disp.max()), float(diff_res.max()), 1e-8)

    frame_indices = [idx for idx in frame_indices if 0 <= idx < ref_video.shape[0]]
    if not frame_indices:
        raise ValueError("No valid frame indices remain for x0 look-ahead figure.")

    n_rows = len(frame_indices)
    n_cols = 7
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 2.2 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "True forward x0",
        "Raw backward candidate",
        "Simple-vector projection",
        "Residual-axis projection",
        "|raw - forward|",
        "|simple - forward|",
        "|residual - forward|",
    ]
    for col in range(n_cols):
        axes[0, col].set_title(col_titles[col], fontsize=10)

    for row_idx, frame_idx in enumerate(frame_indices):
        axes[row_idx, 0].imshow(ref_video[frame_idx].permute(1, 2, 0).numpy())
        axes[row_idx, 1].imshow(raw_video[frame_idx].permute(1, 2, 0).numpy())
        axes[row_idx, 2].imshow(disp_video[frame_idx].permute(1, 2, 0).numpy())
        axes[row_idx, 3].imshow(res_video[frame_idx].permute(1, 2, 0).numpy())
        axes[row_idx, 4].imshow(diff_raw[frame_idx].numpy(), cmap="magma", vmin=0.0, vmax=diff_max)
        axes[row_idx, 5].imshow(diff_disp[frame_idx].numpy(), cmap="magma", vmin=0.0, vmax=diff_max)
        axes[row_idx, 6].imshow(diff_res[frame_idx].numpy(), cmap="magma", vmin=0.0, vmax=diff_max)
        for col in range(n_cols):
            axes[row_idx, col].set_xticks([])
            axes[row_idx, col].set_yticks([])
        axes[row_idx, 0].set_ylabel(f"Frame {frame_idx}", fontsize=10)

    suptitle = (
        f"One-step x0 look-ahead from the SAME noisy state | case={snap['case_name']} active={snap['active_step_rank']} loop={snap['loop_step']}\n"
        f"Motion-region local cosine to true forward: raw={float(snap['motion_mean_cos_raw']):.3f}, simple={float(snap['motion_mean_cos_disp']):.3f}, residual={float(snap['motion_mean_cos_res']):.3f}"
    )
    fig.suptitle(suptitle, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render true-forward rebuttal figures for MR-CATR.")
    parser.add_argument("--analysis_root", type=str, required=True)
    parser.add_argument("--fig_root", type=str, required=True)
    parser.add_argument("--case_filter", type=str, default="")
    parser.add_argument("--representative_case", type=str, default="")
    parser.add_argument("--representative_step", type=int, default=-1)
    parser.add_argument("--frame_indices", type=str, default="auto")
    parser.add_argument("--make_global", action="store_true")
    parser.add_argument("--make_dual_plane", action="store_true")
    parser.add_argument("--make_x0", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--config", type=str, default="scripts/sampling/configs/svd_xt.yaml")
    parser.add_argument("--num_frames", type=int, default=25)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--version", type=str, default="svd_xt")
    parser.add_argument("--decoding_t", type=int, default=2)
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    if not (args.make_global or args.make_dual_plane or args.make_x0):
        args.make_global = True
        args.make_dual_plane = True
        args.make_x0 = True

    analysis_root = Path(args.analysis_root)
    fig_root = ensure_dir(Path(args.fig_root))
    case_filter = parse_csv_list(args.case_filter)

    case_dirs = find_case_dirs(analysis_root, case_filter)
    if not case_dirs:
        raise ValueError(f"No case folders with metrics.csv were found under {analysis_root}")

    all_rows = collect_all_metrics(case_dirs)
    case_ranking_rows, snapshot_ranking_rows, summary_text = build_rankings(all_rows)
    global_dir = ensure_dir(fig_root / "global")
    save_text(summary_text, global_dir / "summary_true_forward.txt")
    save_csv(case_ranking_rows, global_dir / "case_ranking_true_forward.csv")
    save_csv(snapshot_ranking_rows, global_dir / "snapshot_ranking_true_forward.csv")

    if args.make_global:
        plot_global_true_forward_alignment(all_rows, global_dir / "fig_global_true_forward_alignment.png")

    rep_case = args.representative_case
    rep_step = None if args.representative_step < 0 else int(args.representative_step)
    case_name, snapshot_path = choose_snapshot_path(case_dirs, snapshot_ranking_rows, rep_case, rep_step)
    case_dir = analysis_root / case_name

    if args.make_dual_plane:
        plot_dual_plane_alignment(snapshot_path, fig_root / case_name / f"fig_dual_plane_true_forward_{snapshot_path.stem}.png")

    if args.make_x0:
        model, _ = lazy_load_model(
            config=args.config,
            device=args.device,
            num_frames=args.num_frames,
            num_steps=args.num_steps,
            verbose=False,
            with_filter=False,
        )
        snap = torch.load(snapshot_path, map_location="cpu")
        frame_indices = parse_frame_indices(args.frame_indices, int(snap["num_frames"]))
        plot_x0_lookahead(
            snapshot_path=snapshot_path,
            case_dir=case_dir,
            out_path=fig_root / case_name / f"fig_x0_lookahead_true_forward_{snapshot_path.stem}.png",
            model=model,
            device=args.device,
            decoding_t=args.decoding_t,
            frame_indices=frame_indices,
        )

    best = snapshot_ranking_rows[0] if snapshot_ranking_rows else None
    if best is not None:
        msg = (
            f"Best snapshot by residual-over-simple alignment gain: case={best['case_name']} active_step={best['active_step_rank']} "
            f"loop={best['loop_step']} score={best['score']:.4f}"
        )
        print(msg)
    print(f"Figures saved under: {fig_root}")
    print(f"Summary: {global_dir / 'summary_true_forward.txt'}")
    print(f"Snapshot ranking: {global_dir / 'snapshot_ranking_true_forward.csv'}")


if __name__ == "__main__":
    main()
