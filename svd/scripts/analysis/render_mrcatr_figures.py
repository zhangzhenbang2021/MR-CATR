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





def _dot(a, b, eps: float = 1e-8):
    return torch.sum(a.float() * b.float())


def _norm(a, eps: float = 1e-8):
    return torch.sqrt(torch.sum(a.float() * a.float()) + eps)


def _normalize(v, eps: float = 1e-8):
    return v / (_norm(v, eps=eps) + eps)


def lazy_load_model(*args, **kwargs):
    from scripts.sampling.mrcatr_rebuttal_diagnostics import load_model
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
                elif key in {"loop_step", "active_step_rank", "positive_before", "positive_after"}:
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
# Diagnostics discovery
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


def choose_snapshot_paths(case_dir: Path, strategy: str, explicit_steps: Sequence[int]) -> List[Path]:
    snapshots = list_snapshot_paths(case_dir)
    if not snapshots:
        return []
    if explicit_steps:
        selected = []
        for step in explicit_steps:
            path = case_dir / "snapshots" / f"step_{int(step):03d}.pt"
            if path.exists():
                selected.append(path)
        return selected
    strategy = strategy.lower()
    if strategy == "all":
        return snapshots
    if strategy == "first":
        return [snapshots[0]]
    if strategy == "last":
        return [snapshots[-1]]
    if strategy == "middle":
        return [snapshots[len(snapshots) // 2]]
    raise ValueError(f"Unknown snapshot strategy: {strategy}")


# -----------------------------------------------------------------------------
# Summary statistics and case ranking
# -----------------------------------------------------------------------------

def aggregate_by_active_step(rows: Sequence[Dict]) -> Dict[int, Dict[str, np.ndarray]]:
    grouped: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        k = int(row["active_step_rank"])
        for metric in [
            "orth_ratio",
            "beta",
            "s_prior",
            "s_agree",
            "effective_correction_ratio",
            "axis_support_ratio",
            "cos_bwd_axis",
            "cos_fused_axis",
        ]:
            grouped[k][metric].append(float(row[metric]))

    result = {}
    for step, metrics in grouped.items():
        result[step] = {name: np.asarray(values, dtype=np.float64) for name, values in metrics.items()}
    return result

def compute_global_summary(rows: Sequence[Dict]) -> Tuple[str, List[Dict]]:
    if not rows:
        return "No metrics rows were found.\n", []

    orth = np.asarray([row["orth_ratio"] for row in rows], dtype=np.float64)
    beta = np.asarray([row["beta"] for row in rows], dtype=np.float64)
    s_prior = np.asarray([row["s_prior"] for row in rows], dtype=np.float64)
    s_agree = np.asarray([row["s_agree"] for row in rows], dtype=np.float64)
    eff_corr = np.asarray([row["effective_correction_ratio"] for row in rows], dtype=np.float64)
    axis_support = np.asarray([row["axis_support_ratio"] for row in rows], dtype=np.float64)
    cos_bwd_axis = np.asarray([row["cos_bwd_axis"] for row in rows], dtype=np.float64)
    cos_fused_axis = np.asarray([row["cos_fused_axis"] for row in rows], dtype=np.float64)
    eff_signal = beta * orth

    case_groups: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        case_groups[row["case_name"]].append(row)

    ranking_rows: List[Dict] = []
    for case_name, case_rows in sorted(case_groups.items()):
        case_beta = np.asarray([r["beta"] for r in case_rows], dtype=np.float64)
        case_orth = np.asarray([r["orth_ratio"] for r in case_rows], dtype=np.float64)
        case_eff = np.asarray([r["effective_correction_ratio"] for r in case_rows], dtype=np.float64)
        case_s_prior = np.asarray([r["s_prior"] for r in case_rows], dtype=np.float64)
        case_axis = np.asarray([r["axis_support_ratio"] for r in case_rows], dtype=np.float64)
        ranking_rows.append(
            {
                "case_name": case_name,
                "n_steps": int(len(case_rows)),
                "mean_beta": float(case_beta.mean()),
                "mean_orth_ratio": float(case_orth.mean()),
                "mean_effective_correction_ratio": float(case_eff.mean()),
                "mean_effective_signal": float((case_beta * case_orth).mean()),
                "mean_s_prior": float(case_s_prior.mean()),
                "positive_s_prior_rate": float(100.0 * np.mean(case_s_prior > 0.0)),
                "mean_axis_support_ratio": float(case_axis.mean()),
            }
        )
    ranking_rows.sort(key=lambda row: row["mean_effective_signal"], reverse=True)

    text = []
    text.append("MR-CATR rebuttal visualization summary\n")
    text.append(f"Cases analyzed: {len(case_groups)}")
    text.append(f"Active-step records analyzed: {len(rows)}")
    text.append(f"Mean beta: {beta.mean():.4f}")
    text.append(f"Mean removed orthogonal ratio: {orth.mean():.4f}")
    text.append(f"Mean axis support ratio: {axis_support.mean():.4f}")
    text.append(f"Mean effective correction ratio: {eff_corr.mean():.4f}")
    text.append(f"Mean effective signal beta*orth: {eff_signal.mean():.4f}")
    text.append(f"Mean s_prior: {s_prior.mean():.4f}")
    text.append(f"Positive s_prior rate: {100.0 * np.mean(s_prior > 0.0):.2f}%")
    text.append(f"Mean s_agree: {s_agree.mean():.4f}")
    text.append(f"Mean cos(raw, axis): {cos_bwd_axis.mean():.4f}")
    text.append(f"Mean cos(fused, axis): {cos_fused_axis.mean():.4f}")
    text.append("")
    text.append("Per-case ranking by effective signal beta*orth:")
    for row in ranking_rows:
        text.append(
            f"  - {row['case_name']}: eff_signal={row['mean_effective_signal']:.4f}, "
            f"beta={row['mean_beta']:.4f}, orth={row['mean_orth_ratio']:.4f}, "
            f"eff_corr={row['mean_effective_correction_ratio']:.4f}, s_prior={row['mean_s_prior']:.4f}"
        )
    return "\n".join(text) + "\n", ranking_rows

def plot_stepwise_alignment_curves(rows: Sequence[Dict], out_path: Path):
    grouped = aggregate_by_active_step(rows)
    if not grouped:
        raise ValueError("No step-wise metrics were found. Cannot draw alignment curves.")

    steps = sorted(grouped.keys())
    beta_mean = [grouped[s]["beta"].mean() for s in steps]
    beta_std = [grouped[s]["beta"].std() for s in steps]
    eff_mean = [grouped[s]["effective_correction_ratio"].mean() for s in steps]
    eff_std = [grouped[s]["effective_correction_ratio"].std() for s in steps]
    orth_mean = [grouped[s]["orth_ratio"].mean() for s in steps]
    orth_std = [grouped[s]["orth_ratio"].std() for s in steps]
    s_prior_mean = [grouped[s]["s_prior"].mean() for s in steps]
    s_prior_std = [grouped[s]["s_prior"].std() for s in steps]

    plt.figure(figsize=(8, 12))

    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(steps, beta_mean, marker="o", label="beta gate")
    ax1.fill_between(steps, np.asarray(beta_mean) - np.asarray(beta_std), np.asarray(beta_mean) + np.asarray(beta_std), alpha=0.2)
    ax1.set_ylabel("beta")
    ax1.set_title("Step-wise gated-correction statistics in MR-CATR active steps")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(4, 1, 2)
    ax2.plot(steps, eff_mean, marker="o", label="Effective correction ratio")
    ax2.fill_between(steps, np.asarray(eff_mean) - np.asarray(eff_std), np.asarray(eff_mean) + np.asarray(eff_std), alpha=0.2)
    ax2.set_ylabel(r"||d_fused - d_bwd|| / ||d_bwd||")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = plt.subplot(4, 1, 3)
    ax3.plot(steps, orth_mean, marker="o", label="Removed orthogonal ratio")
    ax3.fill_between(steps, np.asarray(orth_mean) - np.asarray(orth_std), np.asarray(orth_mean) + np.asarray(orth_std), alpha=0.2)
    ax3.set_ylabel(r"||d_perp|| / ||d_bwd||")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4 = plt.subplot(4, 1, 4)
    ax4.plot(steps, s_prior_mean, marker="o", label="s_prior")
    ax4.fill_between(steps, np.asarray(s_prior_mean) - np.asarray(s_prior_std), np.asarray(s_prior_mean) + np.asarray(s_prior_std), alpha=0.2)
    ax4.set_xlabel("Active-step rank")
    ax4.set_ylabel("cos(m_start, m_end)")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def tensor_to_coords(v: torch.Tensor, e1: torch.Tensor, e2: torch.Tensor, denom: float) -> Tuple[float, float]:
    x = float(_dot(v, e1).detach().cpu().item()) / denom
    y = float(_dot(v, e2).detach().cpu().item()) / denom
    return x, y


def choose_second_basis(d_bwd: torch.Tensor, e1: torch.Tensor, fallback_vec: torch.Tensor) -> torch.Tensor:
    d_perp = d_bwd - _dot(d_bwd, e1) * e1
    if float(_norm(d_perp).detach().cpu().item()) > 1e-8:
        return _normalize(d_perp)

    alt = fallback_vec - _dot(fallback_vec, e1) * e1
    if float(_norm(alt).detach().cpu().item()) > 1e-8:
        return _normalize(alt)

    flat = torch.zeros_like(d_bwd)
    flat.view(-1)[0] = 1.0
    alt = flat - _dot(flat, e1) * e1
    return _normalize(alt)

def plot_exact_2d_plane(snapshot_path: Path, out_path: Path):
    snapshot = torch.load(snapshot_path, map_location="cpu")
    d_bwd = snapshot["d_bwd"].float()
    d_bwd_aligned = snapshot["d_bwd_aligned"].float()
    d_fused = snapshot["d_fused"].float() if "d_fused" in snapshot else snapshot["d_bwd_aligned"].float()
    m = snapshot["m"].float()

    e1 = _normalize(m)
    e2 = choose_second_basis(d_bwd, e1, d_fused)
    denom = float(_norm(d_bwd).detach().cpu().item()) + 1e-8

    raw_xy = tensor_to_coords(d_bwd, e1, e2, denom)
    aligned_xy = tensor_to_coords(d_bwd_aligned, e1, e2, denom)
    fused_xy = tensor_to_coords(d_fused, e1, e2, denom)
    proj_xy = (raw_xy[0], 0.0)

    points = np.asarray([[0.0, 0.0], raw_xy, aligned_xy, fused_xy, proj_xy], dtype=np.float64)
    max_abs = max(1.0, float(np.max(np.abs(points))) * 1.25)

    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    ax.axhline(0.0, linewidth=1.0, alpha=0.5)
    ax.axvline(0.0, linewidth=1.0, alpha=0.3)

    arrow_specs = [
        (raw_xy, "Raw backward", "tab:blue"),
        (aligned_xy, "Aligned candidate", "tab:orange"),
        (fused_xy, "Fused actual update", "tab:purple"),
    ]
    for xy, label, color in arrow_specs:
        ax.annotate(
            "",
            xy=xy,
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", linewidth=2.5, color=color),
        )
        ax.scatter([xy[0]], [xy[1]], s=70, color=color, label=label, zorder=3)

    ax.plot([0.0, proj_xy[0]], [0.0, proj_xy[1]], linestyle="--", linewidth=1.5, color="gray")
    ax.plot([proj_xy[0], raw_xy[0]], [proj_xy[1], raw_xy[1]], linestyle="--", linewidth=1.5, color="gray")
    ax.scatter([proj_xy[0]], [proj_xy[1]], s=45, color="black", zorder=3)

    xoff = 0.03 * max_abs
    yoff = 0.03 * max_abs
    ax.text(proj_xy[0] + xoff, proj_xy[1] - yoff, "projection point", fontsize=9, va="top")

    axis_support_ratio = float(_norm(d_bwd_aligned).detach().cpu().item()) / denom
    orth_ratio = float(abs(raw_xy[1]))
    effective_correction_ratio = float(_norm(d_fused - d_bwd).detach().cpu().item()) / denom
    beta = float(snapshot.get("beta", 0.0))
    s_prior = float(snapshot.get("s_prior", 0.0))

    info_text = (
        f"case = {snapshot['case_name']}\n"
        f"active step = {snapshot['active_step_rank']}\n"
        f"loop step = {snapshot['loop_step']}\n"
        f"axis support ratio = {axis_support_ratio:.3f}\n"
        f"removed orth ratio = {orth_ratio:.3f}\n"
        f"beta = {beta:.3f}\n"
        f"effective correction = {effective_correction_ratio:.3f}\n"
        f"s_prior = {s_prior:.3f}"
    )
    ax.text(0.03, 0.97, info_text, transform=ax.transAxes, va="top", ha="left", fontsize=10, bbox=dict(boxstyle="round", alpha=0.08))

    ax.set_xlim(-max_abs, max_abs)
    ax.set_ylim(-max_abs, max_abs)
    ax.set_xlabel(r"Coordinate along motion axis $m_t$ (normalized by $||d_{bwd}||$)")
    ax.set_ylabel(r"Coordinate along orthogonal conflict direction (normalized by $||d_{bwd}||$)")
    ax.set_title("Exact 2D decomposition plane of real high-dimensional latent updates")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", framealpha=0.9)

    ensure_dir(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

def decode_latents(model, latents: torch.Tensor, device: str, decoding_t: int) -> torch.Tensor:
    model.en_and_decode_n_samples_a_time = decoding_t
    model = model.to(torch.float32)
    latents = latents.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        samples_x = model.decode_first_stage(latents)
    samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).detach().cpu()
    model = model.to(torch.float16)
    return samples


def tensor_to_uint8_video(video: torch.Tensor) -> np.ndarray:
    video_np = video.permute(0, 2, 3, 1).numpy()
    return np.clip(video_np * 255.0, 0, 255).astype(np.uint8)


def save_gif(video: torch.Tensor, path: Path, duration_ms: int = 125):
    frames = tensor_to_uint8_video(video)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)


def build_heatmap_overlay(base_video: torch.Tensor, d_perp: torch.Tensor) -> torch.Tensor:
    base_video = base_video.clone()
    heat = torch.sqrt(torch.sum(d_perp.float() ** 2, dim=1, keepdim=True))
    heat = F.interpolate(heat, size=base_video.shape[-2:], mode="bilinear", align_corners=False)
    heat = heat.squeeze(1)
    heat_np = heat.numpy()
    scale = float(np.quantile(heat_np, 0.995))
    scale = max(scale, 1e-8)
    heat = torch.clamp(heat / scale, 0.0, 1.0)

    overlays = []
    cmap = plt.get_cmap("inferno")
    for idx in range(heat.shape[0]):
        color = torch.from_numpy(cmap(heat[idx].numpy())[:, :, :3]).permute(2, 0, 1).float()
        overlay = 0.6 * base_video[idx] + 0.4 * color
        overlays.append(torch.clamp(overlay, 0.0, 1.0))
    return torch.stack(overlays, dim=0), heat


def plot_pixel_effects(
    snapshot_path: Path,
    out_path: Path,
    model,
    device: str,
    decoding_t: int,
    frame_indices: Sequence[int],
    make_gifs: bool,
):
    snapshot = torch.load(snapshot_path, map_location="cpu")
    x_base = snapshot["x_base"].float()
    d_bwd = snapshot["d_bwd"].float()
    d_fused = snapshot["d_fused"].float() if "d_fused" in snapshot else snapshot["d_bwd_aligned"].float()
    d_perp = snapshot["d_perp"].float()

    x_raw = x_base + d_bwd
    x_fused = x_base + d_fused

    base_video = decode_latents(model, x_base, device=device, decoding_t=decoding_t)
    raw_video = decode_latents(model, x_raw, device=device, decoding_t=decoding_t)
    fused_video = decode_latents(model, x_fused, device=device, decoding_t=decoding_t)
    overlay_video, heat_scalar = build_heatmap_overlay(base_video, d_perp)

    diff_raw = torch.mean(torch.abs(raw_video - base_video), dim=1)
    diff_effective = torch.mean(torch.abs(fused_video - raw_video), dim=1)

    frame_indices = [idx for idx in frame_indices if 0 <= idx < base_video.shape[0]]
    if not frame_indices:
        raise ValueError("No valid frame indices remain for pixel-effect visualization.")

    n_rows = len(frame_indices)
    n_cols = 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.2 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = [
        "Current state x_base",
        "After raw backward step",
        "After fused actual step",
        "|raw - current|",
        "|fused - raw|",
        "Removed orthogonal component",
    ]
    for col in range(n_cols):
        axes[0, col].set_title(col_titles[col], fontsize=10)

    diff_max = max(float(diff_raw.max().item()), float(diff_effective.max().item()), 1e-8)

    for row_idx, frame_idx in enumerate(frame_indices):
        axes[row_idx, 0].imshow(base_video[frame_idx].permute(1, 2, 0).numpy())
        axes[row_idx, 1].imshow(raw_video[frame_idx].permute(1, 2, 0).numpy())
        axes[row_idx, 2].imshow(fused_video[frame_idx].permute(1, 2, 0).numpy())
        axes[row_idx, 3].imshow(diff_raw[frame_idx].numpy(), cmap="magma", vmin=0.0, vmax=diff_max)
        axes[row_idx, 4].imshow(diff_effective[frame_idx].numpy(), cmap="magma", vmin=0.0, vmax=diff_max)
        axes[row_idx, 5].imshow(overlay_video[frame_idx].permute(1, 2, 0).numpy())
        axes[row_idx, 5].contour(heat_scalar[frame_idx].numpy(), levels=5, linewidths=0.4)

        for col in range(n_cols):
            axes[row_idx, col].set_xticks([])
            axes[row_idx, col].set_yticks([])
        axes[row_idx, 0].set_ylabel(f"Frame {frame_idx}", fontsize=10)

    fig.suptitle(
        f"One-step decoded effect at case={snapshot['case_name']}, active-step={snapshot['active_step_rank']}",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    if make_gifs:
        gif_root = ensure_dir(out_path.parent / "gifs")
        stem = out_path.stem
        save_gif(base_video, gif_root / f"{stem}_current.gif")
        save_gif(raw_video, gif_root / f"{stem}_raw.gif")
        save_gif(fused_video, gif_root / f"{stem}_fused.gif")
        save_gif(overlay_video, gif_root / f"{stem}_orth_overlay.gif")

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the three MR-CATR rebuttal visualization figures from saved diagnostics.")
    parser.add_argument("--analysis_root", type=str, required=True, help="Folder produced by mrcatr_rebuttal_diagnostics.py")
    parser.add_argument("--fig_root", type=str, required=True, help="Folder where PNG/PDF figures will be saved")
    parser.add_argument("--case_filter", type=str, default="", help="Optional comma-separated list of case names to render")
    parser.add_argument("--snapshot_strategy", type=str, default="middle", choices=["first", "middle", "last", "all"], help="How to pick snapshots per case if --snapshot_steps is not given")
    parser.add_argument("--snapshot_steps", type=str, default="", help="Optional comma-separated active-step ranks, e.g. 0,2,4")
    parser.add_argument("--frame_indices", type=str, default="auto", help="Frames to show in the pixel-effect figure. Default auto -> 5 evenly spaced frames")
    parser.add_argument("--make_fig1", action="store_true", help="Render exact 2D decomposition-plane figures")
    parser.add_argument("--make_fig2", action="store_true", help="Render dataset-level step-wise alignment curves")
    parser.add_argument("--make_fig3", action="store_true", help="Render pixel-space one-step effect figures (requires decoder)")
    parser.add_argument("--make_gifs", action="store_true", help="When rendering fig3, also save GIFs for current/raw/aligned/orth-overlay sequences")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--config", type=str, default="scripts/sampling/configs/svd_xt.yaml")
    parser.add_argument("--num_frames", type=int, default=25)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--version", type=str, default="svd_xt")
    parser.add_argument("--decoding_t", type=int, default=4)
    return parser


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    if not (args.make_fig1 or args.make_fig2 or args.make_fig3):
        args.make_fig1 = True
        args.make_fig2 = True
        args.make_fig3 = True

    analysis_root = Path(args.analysis_root)
    fig_root = ensure_dir(Path(args.fig_root))
    case_filter = parse_csv_list(args.case_filter)
    explicit_snapshot_steps = [int(item) for item in parse_csv_list(args.snapshot_steps)]

    case_dirs = find_case_dirs(analysis_root, case_filter)
    if not case_dirs:
        raise ValueError(f"No case folders with metrics.csv were found under {analysis_root}")

    all_rows = collect_all_metrics(case_dirs)
    summary_text, ranking_rows = compute_global_summary(all_rows)
    ensure_dir(fig_root / "global")
    save_text(summary_text, fig_root / "global" / "summary_global.txt")
    save_csv(ranking_rows, fig_root / "global" / "case_ranking.csv")

    if args.make_fig2:
        plot_stepwise_alignment_curves(all_rows, fig_root / "global" / "fig2_stepwise_alignment_curves.png")

    selected_snapshot_paths: List[Tuple[str, Path]] = []
    for case_dir in case_dirs:
        for snap_path in choose_snapshot_paths(case_dir, args.snapshot_strategy, explicit_snapshot_steps):
            selected_snapshot_paths.append((case_dir.name, snap_path))

    if args.make_fig1:
        for case_name, snap_path in selected_snapshot_paths:
            out_path = fig_root / case_name / f"fig1_exact_2d_plane_{snap_path.stem}.png"
            plot_exact_2d_plane(snap_path, out_path)

    if args.make_fig3:
        model, _ = lazy_load_model(
            config=args.config,
            device=args.device,
            num_frames=args.num_frames,
            num_steps=args.num_steps,
            verbose=False,
            with_filter=False,
        )
        for case_name, snap_path in selected_snapshot_paths:
            snapshot = torch.load(snap_path, map_location="cpu")
            frame_indices = parse_frame_indices(args.frame_indices, int(snapshot["num_frames"]))
            out_path = fig_root / case_name / f"fig3_pixel_effect_{snap_path.stem}.png"
            plot_pixel_effects(
                snapshot_path=snap_path,
                out_path=out_path,
                model=model,
                device=args.device,
                decoding_t=args.decoding_t,
                frame_indices=frame_indices,
                make_gifs=args.make_gifs,
            )

    print(f"Figures saved under: {fig_root}")
    print(f"Global summary: {fig_root / 'global' / 'summary_global.txt'}")
    print(f"Case ranking CSV: {fig_root / 'global' / 'case_ranking.csv'}")


if __name__ == "__main__":
    main()
