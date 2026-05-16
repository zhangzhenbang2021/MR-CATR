import math
import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))
import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config, append_dims
from torchvision.transforms import ToTensor
from tqdm import tqdm


def _dot(a, b, eps=1e-8):
    # a,b: (T,C,H,W)
    return torch.sum(a.float() * b.float())

def _norm(a, eps=1e-8):
    return torch.sqrt(torch.sum(a.float() * a.float()) + eps)

def _normalize(v, eps=1e-8):
    return v / (_norm(v, eps=eps) + eps)

def _cos(a, b, eps=1e-8):
    return _dot(a, b, eps=eps) / (_norm(a, eps=eps) * _norm(b, eps=eps) + eps)


def motion_axis_from_denoised(denoised, eps=1e-8):

    res = denoised[1:] - denoised[:-1]       
    m = torch.mean(res, dim=0)                
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
    # projection onto span(m)
    coeff = _dot(d_bwd, m, eps=eps)  # scalar
    return coeff * m                 # (T,C,H,W)

def rho_schedule(sigma, sigma_T, p=2.0):
    # sigma, sigma_T are scalars (torch tensors)
    return 1.0 - (sigma / sigma_T).clamp(min=0.0, max=1.0) ** p

def conflict_aware_fuse(d_bwd, d_bwd_aligned, sigma, sigma_T, p=2.0, eps=1e-8):
    s = _cos(d_bwd, d_bwd_aligned, eps=eps)
    s = torch.clamp(s, min=0.0)  # tensor scalar

    rho = rho_schedule(sigma, sigma_T, p=p)     # tensor scalar
    beta = rho * s                              # tensor scalar

    d = (1 - beta) * d_bwd + beta * d_bwd_aligned
    return d, beta, s, rho



def sample(

    input_start_path: str = "", 
    input_end_path: str = "", 
     
    
    num_frames: Optional[int] = None,  # 21 for SV3D
    num_steps: Optional[int] = None,
    version: str = "svd_xt",
    fps_id: int = 10,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 4,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    cfg_scale: float = 1.0,
    cfg_scale_flip: float = 1.0,
    output_folder: Optional[str] = None,
    verbose: Optional[bool] = False,
):
    """
    If you run out of VRAM, try decreasing `decoding_t`.
    """
    if version == "svd_xt":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 25)
        model_config = "scripts/sampling/configs/svd_xt.yaml"
    else:
        raise ValueError(f"Version {version} does not exist.")

    model, filter = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
        verbose,
    )
    torch.manual_seed(seed)

    with Image.open(input_start_path) as image:
        input_image = image.convert("RGB")
        input_image = input_image.resize((1024, 576))
        w, h = input_image.size

        if h % 64 != 0 or w % 64 != 0:
            width, height = map(lambda x: x - x % 64, (w, h))
            input_image = input_image.resize((width, height))
            print(
                f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
            )

    image = ToTensor()(input_image)
    image = image * 2.0 - 1.0
    image = image.unsqueeze(0).to(device).to(torch.float16)
    latent = model.encode_first_stage(image)

    ## load end frame
    input_image_end = Image.open(input_end_path).convert("RGB").resize((1024, 576))
    image_end = ToTensor()(input_image_end)
    image_end = image_end * 2.0 - 1.0
    image_end = image_end.unsqueeze(0).to(device).to(torch.float16)
    latent_end = model.encode_first_stage(image_end)
    ##

    H, W = image.shape[2:]
    assert image.shape[1] == 3
    F = 8
    C = 4
    shape = (num_frames, C, H // F, W // F)
    if motion_bucket_id > 255:
        print(
            "WARNING: High motion bucket! This may lead to suboptimal performance."
        )

    if fps_id < 5:
        print("WARNING: Small fps value! This may lead to suboptimal performance.")

    if fps_id > 30:
        print("WARNING: Large fps value! This may lead to suboptimal performance.")

    value_dict = {}
    value_dict["cond_frames_without_noise"] = image
    value_dict["motion_bucket_id"] = motion_bucket_id
    value_dict["fps_id"] = fps_id
    value_dict["cond_aug"] = cond_aug
    value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)

    ## symmetric condition
    value_dict_end = {}
    value_dict_end["cond_frames_without_noise"] = image_end
    value_dict_end["motion_bucket_id"] = motion_bucket_id
    value_dict_end["fps_id"] = fps_id
    value_dict_end["cond_aug"] = cond_aug
    value_dict_end["cond_frames"] = image_end + cond_aug * torch.randn_like(image_end)
    ##

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

            ## symmetric condition
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
            ##

            randn = torch.randn(shape, device=device)

            additional_model_inputs = {}
            additional_model_inputs["image_only_indicator"] = torch.zeros(
                2, num_frames
            ).to(device)
            additional_model_inputs["num_video_frames"] = batch["num_video_frames"]
            
            def denoiser(x, sigma, c, uc):
                c_out = dict()
                for k in c:
                    if k in ["vector", "crossattn", "concat"]:
                        c_out[k] = torch.cat((uc[k], c[k]), 0)
                    else:
                        assert c[k] == uc[k]
                        c_out[k] = c[k]
                denoiser_input, denoiser_sigma, denoiser_c = torch.cat([x] * 2), torch.cat([sigma] * 2), c_out
                sigma_shape = denoiser_sigma.shape
                denoiser_sigma = append_dims(denoiser_sigma, x.ndim)
                c_skip = 1.0 / (denoiser_sigma**2 + 1.0)
                c_out = -denoiser_sigma / (denoiser_sigma**2 + 1.0) ** 0.5
                c_in = 1.0 / (denoiser_sigma**2 + 1.0) ** 0.5
                c_noise = 0.25 * denoiser_sigma.log()
                c_noise = c_noise.reshape(sigma_shape)
                ## Denoise
                denoised = model.model(denoiser_input * c_in, c_noise, denoiser_c, **additional_model_inputs) * c_out + denoiser_input * c_skip
                ## CFG++ guidance
                x_u, x_c = denoised.chunk(2)
                return x_u, x_c
            
            def CFG(x_u, x_c, scale):
                x_u = rearrange(x_u, "(b t) ... -> b t ...", t=num_frames)
                x_c = rearrange(x_c, "(b t) ... -> b t ...", t=num_frames)
                scale = torch.linspace(scale, scale, steps=num_frames).unsqueeze(0)
                scale = repeat(scale, "1 t -> b t", b=x_u.shape[0])
                scale = append_dims(scale, x_u.ndim).to(x_u.device)
                denoised =  rearrange(x_u + scale * (x_c - x_u), "b t ... -> (b t) ...")
                return denoised

            def masking(x, index):
                mask = torch.zeros_like(x)  # Initialize a mask tensor of zeros with the same shape
                mask[index, :, :, :] = 1
                return x * mask
            
            def CG(A, b, x, n_inner=5, eps=1e-5):
                r = b - A(x)
                p = r.clone()
                rsold = torch.sum(r * r, dim=[0, 1, 2, 3], keepdim=True)  # (b, 1, 1, 1, 1)#rsold = th.matmul(r.view(1, -1), r.view(1, -1).T)
                for i in range(n_inner):
                    Ap = A(p)
                    a = rsold / torch.sum(p * Ap, dim=[0, 1, 2, 3], keepdim=True)  # (b, 1, 1, 1, 1)#a = rsold / th.matmul(p.view(1, -1), Ap.view(1, -1).T)
                    x = x + a * p
                    r = r - a * Ap
                    rsnew = torch.sum(r * r, dim=[0, 1, 2, 3], keepdim=True)  # (b, 1, 1, 1, 1)#rsnew = th.matmul(r.view(1, -1), r.view(1, -1).T)
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
                def Acg(x):
                    return AT(A(x))
                Acg_fn = Acg
                bcg = AT(measurement)
                return CG(Acg_fn, bcg, x, n_inner=n_inner)
            
            x, s_in, sigmas, num_sigmas, cond, uc = model.sampler.prepare_sampling_loop(randn, c, uc, num_steps)
            
            for i in tqdm(model.sampler.get_sigma_gen(num_sigmas), total=num_sigmas-1):
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
                
                
                if i < 0.2 * num_steps:
                    # Forward sample
                    x_u, x_c = denoiser(x, sigma_hat, cond, uc)
                    denoised = CFG(x_u, x_c, scale=cfg_scale)

                    denoised_start_for_axis = denoised.detach()   

                    denoised_hat = DDS(denoised, n_inner=5, latent=latent_end)
                    d = (x - x_u) / append_dims(sigma_hat, x.ndim)
                    dt = append_dims(next_sigma, x.ndim)
                    x = denoised_hat + d * dt

                    eps = torch.randn_like(x) * model.sampler.s_noise
                    x = x + eps * append_dims(sigma_hat**2 - next_sigma**2, x.ndim) ** 0.5

                    x = torch.flip(x, dims=[0])
                    x_base = x 

                    x_u_end, x_c_end = denoiser(x, sigma_hat, c_end, uc_end)
                    denoised_end = CFG(x_u_end, x_c_end, scale=cfg_scale_flip)

                    m_start = motion_axis_from_denoised(
                        torch.flip(denoised_start_for_axis, dims=[0]) 
                    )

                    m_end = motion_axis_from_denoised(denoised_end.detach())
                    m, s_prior = consensus_axis(m_start, m_end, kappa=0.2)

                    denoised_hat = DDS(denoised_end, n_inner=5, latent=latent)

                    d_raw = (x_base - x_u_end) / append_dims(sigma_hat, x.ndim)
                    dt = append_dims(next_sigma, x.ndim)
                    x_raw_next = denoised_hat + d_raw * dt
                    d_bwd = x_raw_next - x_base  

                    d_bwd_aligned = project_update(d_bwd, m)

                    d_fused, beta, s_agree, rho = conflict_aware_fuse(
                        d_bwd, d_bwd_aligned, sigmas[i], sigmas[0], p=2.0
                    )


                    x = x_base + d_fused  

                    x = torch.flip(x, dims=[0])
                else:
                    ### Forward sample ###
                    x_u, x_c = denoiser(x, sigma_hat, cond, uc)
                    denoised = CFG(x_u, x_c, scale=cfg_scale)
                    denoised_hat = DDS(denoised, n_inner=5,latent=latent_end)
                    d = (x - x_u) / append_dims(sigma_hat, x.ndim)
                    dt = append_dims(next_sigma, x.ndim)
                    x = denoised_hat + d * dt
            
                    ### Backward sample ###
                    eps = torch.randn_like(x) * model.sampler.s_noise
                    x = x + eps * append_dims(sigma_hat**2 - next_sigma**2, x.ndim) ** 0.5
                    x = torch.flip(x, dims=[0])
                    x_u, x_c = denoiser(x, sigma_hat, c_end, uc_end)
                    denoised = CFG(x_u, x_c, scale=cfg_scale_flip)
                    denoised_hat = DDS(denoised, n_inner=5,latent=latent)
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
            samples = filter(samples)
            vid = (
                (rearrange(samples, "t c h w -> t h w c") * 255)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            video_path = os.path.join(output_folder, f"{base_count:06d}.gif")

            ## To gif
            images = [Image.fromarray(vid[i]) for i in range(vid.shape[0])]                
            duration = 125              
            images[0].save(video_path, save_all=True, append_images=images[1:], duration=duration, loop=0)


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
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
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.verbose = verbose
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()
    
    model = model.to(torch.float16)

    filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter


if __name__ == "__main__":
    
    
    DATA_PATH = # TODO
    base_output_folder = './output'
    
    cases = [
        {
            "name": "train",
            "start": f"{DATA_PATH}/train/00000.jpg",
            "end":   f"{DATA_PATH}/train/00024.jpg",
        },
        {
            "name": "bear",
            "start": f"{DATA_PATH}/bear/00000.jpg",
            "end":   f"{DATA_PATH}/bear/00024.jpg",
        },
        {
            "name": "boat",
            "start": f"{DATA_PATH}/boat/00000.jpg",
            "end":   f"{DATA_PATH}/boat/00024.jpg",
        },
        {
            "name": "car-turn",
            "start": f"{DATA_PATH}/car-turn/00000.jpg",
            "end":   f"{DATA_PATH}/car-turn/00024.jpg",
        },
    ]
    

    

    for case in cases:
        input_start_path = case["start"]
        input_end_path = case["end"]

        output_folder = f"{base_output_folder}/{case['name']}"

        sample(
            input_start_path=input_start_path,
            input_end_path=input_end_path,
            fps_id=10,
            output_folder=output_folder,
        )
