# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import random
from ruamel import yaml
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
import os
import sys

from utils.misc import load_model_state_dict, str2bool 
sys.path.append('../')
from PIL import Image
import numpy as np
import math
import json 
import argparse


from modelling.diffusion import create_diffusion
from modelling.dit import DiT, DiT_models
from modelling.tokenizer import SoftVQModel, VQ_models


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    # np.savez('gen_imgs.npz', arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = True # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
        
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank
    torch.cuda.set_device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    
    # load VAE
    if args.vq_ckpt is not None:
        print(f"Loading VAE model: {args.vq_ckpt}")
        # create and load model
        vae = VQ_models[args.vq_model](
            image_size=args.image_size,
            codebook_size=args.codebook_size,
            codebook_embed_dim=args.codebook_embed_dim,
            codebook_l2_norm=args.codebook_l2_norm,
            commit_loss_beta=args.commit_loss_beta,
            entropy_loss_ratio=args.entropy_loss_ratio,
            vq_loss_ratio=args.vq_loss_ratio,
            kl_loss_weight=args.kl_loss_weight,
            dropout_p=args.dropout_p,
            enc_type=args.enc_type,
            encoder_model=args.encoder_model,
            dec_type=args.dec_type,
            decoder_model=args.decoder_model,
            num_latent_tokens=args.num_latent_tokens,
            enc_tuning_method=args.encoder_tuning_method,
            dec_tuning_method=args.decoder_tuning_method,
            enc_pretrained=args.encoder_pretrained,
            dec_pretrained=args.decoder_pretrained,
            enc_patch_size=args.encoder_patch_size,
            dec_patch_size=args.decoder_patch_size,
            tau=args.tau,
            repa=args.repa,
            repa_model=args.repa_model,
            repa_patch_size=args.repa_patch_size,
            repa_proj_dim=args.repa_proj_dim,
            repa_loss_weight=args.repa_loss_weight,
            repa_align=args.repa_align,
            num_codebooks=args.num_codebooks,
            # encoder mask modeling
            enc_token_drop=args.enc_token_drop,
            # to pixel
            to_pixel=args.to_pixel,
        )
        if args.vq_ckpt:
            checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
            vae.load_state_dict(load_model_state_dict(checkpoint['model']))
        vae_embed_dim = vae.codebook_embed_dim
        dit_input_size = vae.num_latent_tokens
        vae_mean = args.vq_mean#vae.vq_mean
        vae_std = args.vq_std#vae.vq_std
        vae_string_name = args.vq_ckpt
        vae_1d = True
    else:
        raise NotImplementedError()
    print(f"vae_embed_dim: {vae_embed_dim}, dit_input_size: {dit_input_size}, vae_mean:{vae_mean}, vae_std:{vae_std}, vae_1d: {vae_1d}")
    vae = torch.compile(vae)
    vae = vae.to(device)
    vae.eval()
    
    
    # Load model:
    if args.vq_ckpt is None:
        model = DiT.from_pretrained(f"dit-xl_{args.vq_ckpt}")
        model = torch.compile(model)
        args.model = "DiT-XL/1"
    else:
        model = DiT_models[args.model](
            input_size=dit_input_size,
            in_channels=vae_embed_dim,
            num_classes=args.num_classes,
            vae_1d=vae_1d,
            extra_tokens = args.extra_tokens
        )
        # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
        g_checkpoint = torch.load(args.g_ckpt, map_location="cpu")
        if args.ema:
            state_dict = load_model_state_dict(g_checkpoint['ema'])
        else:
            state_dict = load_model_state_dict(g_checkpoint['model'])
        compiled_state_dict = False 
        for key, item in state_dict.items():
            if '_orig_mod' in key:
                compiled_state_dict = True
                break
        if compiled_state_dict:
            model = torch.compile(model)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        else:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            model = torch.compile(model)
        if missing_keys:
            print(f"[DiT] missing keys in checkpoint: {len(missing_keys)}")
        if unexpected_keys:
            print(f"[DiT] unexpected keys in checkpoint: {len(unexpected_keys)}")
    model = model.to(device)
    model.eval()  # Disable dropout / stochastic depth before sampling.
    diffusion = create_diffusion(str(args.num_sampling_steps), noise_schedule=args.noise_schedule)
    diffusion.vae_1d = vae_1d
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    vae_string_name = vae_string_name.replace("/", "-")
    # folder_name = f"{model_string_name}-{args.noise_schedule}-vae{vae_string_name}-size-{args.image_size}" \
    #               f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    if args.ema:
        folder_name = f"EMA-{args.g_ckpt.split('/')[2]}-{args.g_ckpt.split('/')[-1]}-{args.noise_schedule}-size-{args.image_size}" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    else:
        folder_name = f"{args.g_ckpt.split('/')[2]}-{args.g_ckpt.split('/')[-1]}-{args.noise_schedule}-size-{args.image_size}" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    
    for _ in pbar:
        # Sample inputs:
        # z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        
        with torch.cuda.amp.autocast(dtype=ptdtype): 
            if args.extra_tokens>0:
                z = torch.randn(n, dit_input_size+args.extra_tokens, vae_embed_dim, device=device)
            else:
                z = torch.randn(n, dit_input_size, vae_embed_dim, device=device)        
            y = torch.randint(0, args.num_classes, (n,), device=device)

            # Setup classifier-free guidance:
            if using_cfg:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([args.num_classes] * n, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                sample_fn = model.forward_with_cfg
            else:
                model_kwargs = dict(y=y)
                sample_fn = model.forward

            # Sample images:
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

            samples = vae.decode(samples * vae_std + vae_mean)
            samples = torch.clamp(samples, -1.0, 1.0)
            samples = (samples + 1) / 2
        
        samples = (samples * 255.0).to(torch.uint8)
        # save to images
        samples = samples.permute(0, 2, 3, 1).to("cpu").numpy()

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()
    
    # 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-tokens", type=int, default=0)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--sample-dir", type=str, default="samples")
    
    parser.add_argument("--per-proc-batch-size", type=int, default=64)
    parser.add_argument("--num-fid-samples", type=int, default=3200)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--noise-schedule", type=str, default="linear")
    
    # parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
    #                     help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--g-ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    
    parser.add_argument("--mixed-precision", type=str, default='fp16', choices=["none", "fp16", "bf16"]) 
    
    parser.add_argument("--config", type=str, default='configs/tokenizer/cnn_llamagen_vq16.yaml', help="config file used to specify parameters")
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for eval")
    parser.add_argument("--finetune", type=str2bool, default=False, help="finetune a pre-trained vq model")
    parser.add_argument("--ema", type=str2bool, default=True, help="whether using ema training")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--codebook-l2-norm", type=str2bool, default=True, help="l2 norm codebook")
    parser.add_argument("--codebook-weight", type=float, default=1.0, help="codebook loss weight for vector quantization")
    parser.add_argument("--entropy-loss-ratio", type=float, default=0.0, help="entropy loss ratio in codebook loss")
    parser.add_argument("--vq-loss-ratio", type=float, default=1.0, help="vq loss ratio in codebook loss")
    parser.add_argument("--commit-loss-beta", type=float, default=0.25, help="commit loss beta in codebook loss")
    parser.add_argument("--reconstruction-weight", type=float, default=1.0, help="reconstruction loss weight of image pixel")
    parser.add_argument("--reconstruction-loss", type=str, default='l2', help="reconstruction loss type of image pixel")
    parser.add_argument("--kl-loss-weight", type=float, default=0.000001)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--num-codebooks", type=int, default=1)
    
    parser.add_argument("--perceptual-weight", type=float, default=1.0, help="perceptual loss weight of LPIPS")
    parser.add_argument("--perceptual-loss", type=str, default='vgg', help="perceptual loss type of LPIPS", choices=['vgg', 'timm', 'tv'])
    parser.add_argument("--perceptual-model", type=str, default='vgg', help="perceptual loss model of LPIPS")
    parser.add_argument("--perceptual-dino-variants", type=str, default='depth12_no_train', help="perceptual loss model of LPIPS")
    parser.add_argument("--perceptual-intermediate-loss", type=str2bool, default=False, help="perceptual loss compute at intermedia features of LPIPS")
    parser.add_argument("--perceptual-logit-loss", type=str2bool, default=False, help="perceptual loss compute at logits of LPIPS")
    parser.add_argument("--perceptual-resize", type=str2bool, default=False, help="perceptual loss compute at resized images of LPIPS")
    parser.add_argument("--perceptual-warmup", type=int, default=None, help="iteration to warmup perceptual loss")
    
    parser.add_argument("--disc-weight", type=float, default=0.5, help="discriminator loss weight for gan training")
    parser.add_argument("--disc-start", type=int, default=20000, help="iteration to start discriminator training and loss")
    parser.add_argument("--disc-dim", type=int, default=64, help="discriminator channel base dimension")
    parser.add_argument("--disc-type", type=str, choices=['patchgan', 'stylegan', 'maskbit', 'dino'], default='patchgan', help="discriminator type")
    parser.add_argument("--disc-loss", type=str, choices=['hinge', 'vanilla', 'non-saturating'], default='hinge', help="discriminator loss")
    parser.add_argument("--gen-loss", type=str, choices=['hinge', 'non-saturating'], default='hinge', help="generator loss for gan training")
    parser.add_argument("--lecam-loss-weight", type=float, default=None)
    parser.add_argument("--use-diff-aug",type=str2bool, default=False)
    parser.add_argument("--disc-cr-loss-weight", type=float, default=0.0, help="discriminator consistency loss weight for gan training")
    parser.add_argument("--disc-adaptive-weight",type=str2bool, default=False)
    
    parser.add_argument("--compile", type=str2bool, default=False)
    parser.add_argument("--dropout-p", type=float, default=0.0, help="dropout_p")
    parser.add_argument("--results-dir", type=str, default="results_tokenizer_image")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_epochs", type=int, default=1)
    parser.add_argument("--lr_scheduler", type=str, default='none')
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)

    parser.add_argument("--enc-type", type=str, default="cnn")
    parser.add_argument("--dec-type", type=str, default="cnn")
    parser.add_argument("--num-latent-tokens", type=int, default=None)
    parser.add_argument("--encoder-model", type=str, default='vit_small_patch14_dinov2.lvd142m', help='encoder model name')
    parser.add_argument("--decoder-model", type=str, default='vit_small_patch14_dinov2.lvd142m', help='decoder model name')
    parser.add_argument("--encoder-tuning-method", type=str, default='full', help='tuning method for encoder', choices=['full', 'lora', 'frozen'])
    parser.add_argument("--decoder-tuning-method", type=str, default='full', help='tuning method for decoder', choices=['full', 'lora', 'frozen'])
    parser.add_argument("--encoder-pretrained", type=str2bool, default=True, help='load pre-trained weight for encoder')
    parser.add_argument("--decoder-pretrained", type=str2bool, default=False, help='load pre-trained weight for decoder')
    parser.add_argument("--encoder-patch-size", type=int, default=16, help='encoder patch size')
    parser.add_argument("--decoder-patch-size", type=int, default=16, help='decoder patch size')
    parser.add_argument("--to-pixel", type=str, default="linear")
    
    # repa
    parser.add_argument("--repa", type=str2bool, default=False, help='use repa')
    parser.add_argument('--repa-model', type=str, default='vit_base_patch16_224', help='repa model name')
    parser.add_argument('--repa-patch-size', type=int, default=16, help='repa patch size')
    parser.add_argument('--repa-proj-dim', type=int, default=1024, help='repa embed dim')
    parser.add_argument('--repa-loss-weight', type=float, default=0.1, help='repa loss weight')
    parser.add_argument('--repa-align', type=str, default='global', help='align repa feature', choices=['global', 'avg_1d', 'avg_2d', 'avg_1d_shuffle'])

    # mask modeling
    # make sure drop is 0.0 for not using mask modeling
    parser.add_argument("--enc-token-drop", type=float, default=0.0, help='encoder token drop')
    parser.add_argument("--enc-token-drop-max", type=float, default=0.75, help='maximum drop rate')
    
    
    parser.add_argument('--folder-name', type=str, default=None, help='Name of the experiment folder')
    parser.add_argument('--checkpoints', type=str, default='0010000.pt', help='0010000.pt')
    #fFirst parse of command-line args to check for config file
    args = parser.parse_args()
    
    if args.folder_name is not None:
        base_path = os.path.join('experiments', 'tokenizer', args.folder_name)
        args.config = os.path.join(base_path, 'config.yaml')
        
    # If a config file is specified, load it and set defaults
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**config_args)
    
    # re-parse command-line args to overwrite with any command-line inputs
    args = parser.parse_args()
    main(args)
