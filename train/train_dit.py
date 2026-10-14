# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import json
import random
import ruamel.yaml as yaml
import torch
from tqdm import tqdm
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import wandb
from PIL import Image

from modelling.dit import DiT_models
from modelling.diffusion import create_diffusion
from modelling.tokenizer import VQ_models

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

from utils.data import CachedFolder
from utils.misc import str2bool, manage_checkpoints, load_model_state_dict


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################


# GRAD_NORM_THRESHOLD = 1e2
# SAVE_DIR = "./grad_exploding_debug"
# os.makedirs(SAVE_DIR, exist_ok=True)

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    
    seed = args.global_seed * world_size + rank
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")


    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        if args.exp_index is not None:
            experiment_index = int(args.exp_index)
        else:
            experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        if args.vq_ckpt is None:
            vae_string_name = args.config.split('/')[-2]
        else:
            vae_string_name = args.vq_ckpt.split('/')[-3]+'-'+args.vq_ckpt.split('/')[-1][:-3]
        if args.resume is not None:
            experiment_dir = '/'.join(args.resume.split('/')[:-2])
        else:
            experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-{args.noise_schedule}-{vae_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        os.environ["WANDB_MODE"] = "online"
        wandb_logger = wandb.init(project='DiT_1D', config=vars(args), name=experiment_dir.split('/')[-1])
    else:
        logger = create_logger(None)

    # Create model:
    transform_mean = [0.5, 0.5, 0.5]
    transform_std = [0.5, 0.5, 0.5]
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
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        vae.load_state_dict(load_model_state_dict(checkpoint['model']))
        vae_embed_dim = vae.codebook_embed_dim
        if args.extra_tokens>0:
        else:
            dit_input_size = vae.num_latent_tokens
        vae_mean = args.vq_mean#vae.vq_mean
        vae_std = args.vq_std#vae.vq_std
        vae_string_name = args.vq_ckpt 
        vae_1d = True
    else:
        raise NotImplementedError()
    logger.info(f"vae_embed_dim: {vae_embed_dim}, dit_input_size: {dit_input_size}, vae_mean:{vae_mean}, vae_std:{vae_std}, vae_1d: {vae_1d}")
    vae.eval()
    # assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    model = DiT_models[args.model](
        input_size=dit_input_size,
        in_channels=vae_embed_dim,
        num_classes=args.num_classes,
        vae_1d=vae_1d,
        extra_tokens = args.extra_tokens
    )
    if not args.disable_compile:
        model = torch.compile(model)
        vae = torch.compile(vae)
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    model.vae_1d = model.module.vae_1d
    diffusion = create_diffusion(timestep_respacing="", diffusion_steps=args.train_diffusion_steps, noise_schedule=args.noise_schedule)  # default: 1000 steps, linear noise schedule
    gen_diffusion = create_diffusion(str(250), diffusion_steps=args.train_diffusion_steps, noise_schedule=args.noise_schedule)
    diffusion.vae_1d = vae_1d 
    gen_diffusion.vae_1d = vae_1d
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    for param in vae.parameters():
        param.requires_grad = False
    vae = vae.cuda().eval()
    
    logger.info(f"VAE Parameters: {sum(p.numel() for p in vae.parameters()):,}")
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.module.parameters(), lr=1e-4, weight_decay=0)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=transform_mean, std=transform_std, inplace=True)
    ])
    if args.dataset == 'imagenet':
        dataset = ImageFolder(args.data_path, transform=transform)
    elif args.dataset == 'imagenet_cached':
        dataset = CachedFolder(args.data_path, transform=transform)
    # dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    
    # resume training
    if args.resume is not None:
        logger.info(f"Resumed training from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        # model.module.load_state_dict(checkpoint['model'])
        model.module.load_state_dict(checkpoint['ema'])
        ema.load_state_dict(checkpoint['ema'])
        # opt.load_state_dict(checkpoint['opt'])
        args = checkpoint['args']
        start_epoch = checkpoint['epoch']
        train_steps = checkpoint['train_steps']
    else:
        start_epoch = 0
        train_steps = 0
        

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0
    pbar = tqdm(range(start_epoch, args.epochs), desc="Training epochs") if is_main_process else range(start_epoch, args.epochs)
    
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in pbar:
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for data_batch in loader:
            
            if args.dataset == 'imagenet_cached':
                x, y = data_batch
                x = x.to(device, non_blocking=True)
            elif args.dataset == 'imagenet':
                raw_image, y = data_batch
            
                raw_image = raw_image.to(device, non_blocking=True)

                with torch.no_grad() and torch.cuda.amp.autocast(dtype=ptdtype):
                    # Map input images to latent space + normalize latents:
                    x= vae.encode_images(raw_image)
 
            y = y.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(dtype=ptdtype): 
                    
                x = (x - vae_mean) / vae_std

                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                model_kwargs = dict(y=y)
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
            
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()
                if rank == 0 and wandb_logger is not None:
                    log_dict = {"train_loss": avg_loss}
                    wandb_logger.log(log_dict,
                        step=train_steps
                    )

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        'train_steps': train_steps,
                        'epoch': epoch
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                    # manage checkpoints
                    manage_checkpoints(checkpoint_dir, keep_last_n=10)
                    
                dist.barrier()
                
                # save visualization
            if train_steps % args.vis_every == 0 and train_steps > 0:
                # Sample inputs:
                model.eval()
                n = 8
                if vae_1d:
                    if args.extra_tokens>0:
                        z = torch.randn(n, dit_input_size+args.extra_tokens, vae_embed_dim, device=device)
                    else:
                        z = torch.randn(n, dit_input_size, vae_embed_dim, device=device)
                else:
                    z = torch.randn(n, vae_embed_dim, dit_input_size, dit_input_size, device=device)
                y = torch.randint(0, args.num_classes, (n,), device=device)

                # Setup classifier-free guidance:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([args.num_classes] * n, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=1.0)
                
                sample_fn = model.module.forward_with_cfg

                # Sample images:
                samples = gen_diffusion.p_sample_loop(
                    sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
                )    
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                samples = vae.decode(samples * vae_std + vae_mean)
                
                samples = torch.clamp(samples, -1.0, 1.0)
                samples = (samples + 1) / 2
                grid_sample = make_grid(samples, nrow=8,  padding=2, pad_value=1.0)
                grid_sample = grid_sample.permute(1, 2, 0).mul_(255).cpu().numpy()
                image = Image.fromarray(grid_sample.astype(np.uint8))
                
                if rank == 0:
                    wandb_logger.log({'samples': [wandb.Image(image, caption="generation")]}, step=train_steps)
                model.train()
                dist.barrier()
                    

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()

def manage_checkpoints(save_dir, keep_last_n=2):
    # List all checkpoint files in the save directory
    checkpoints = [f for f in os.listdir(save_dir) if f.endswith('.pt')]
    checkpoints.sort(key=lambda f: int(f.split('/')[-1].split('.')[0]))  # Sort by epoch number

    # If more than `keep_last_n` checkpoints exist, remove the oldest ones
    if len(checkpoints) > keep_last_n + 1:  # keep_last_n + 1 to account for the latest checkpoint
        for checkpoint_file in checkpoints[:-keep_last_n-1]:
            checkpoint_path = os.path.join(save_dir, checkpoint_file)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"Removed old checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-tokens", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="imagenet", choices=["imagenet", "imagenet_cached"])
    parser.add_argument("--data-path", type=str, required=True)
    
    parser.add_argument("--exp-index", type=str, default=None, help="experiment index")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 392, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=512)
    parser.add_argument("--global-seed", type=int, default=0)
    
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--vis-every", type=int, default=5000)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--train-diffusion-steps", type=int, default=1000)
    parser.add_argument("--noise-schedule", type=str, default="linear")
    parser.add_argument("--mixed-precision", type=str, default='none', choices=["none", "fp16", "bf16"]) 
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument("--disable-compile", action='store_true', default=False)
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--vq-mean", type=float, default=0.0)
    parser.add_argument("--vq-std", type=float, default=1.0)
    # vae model
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
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_epochs", type=int, default=1)
    parser.add_argument("--lr_scheduler", type=str, default='none')
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)

    parser.add_argument("--enc-type", type=str, default="cnn")
    parser.add_argument("--dec-type", type=str, default="cnn")
    parser.add_argument("--num-latent-tokens", type=int, default=None)
    parser.add_argument("--specific-rec-loss-weight", type=float, default=0.1, help="specific reconstruction loss weight of image pixel")
    parser.add_argument("--num-generic-latent-tokens", type=int, default=0, help="number of generic latent tokens")
    parser.add_argument("--num-specific-latent-tokens", type=int, default=0, help="number of specific latent tokens")
    parser.add_argument("--mask-prob", type=float, default=0.0, help="mask probability for specific latent tokens")
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
    parser.add_argument('--repa-token', type=str, default='generic', help='repa tokens', choices=['all', 'generic', 'specific'])

    
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
            
            # 不允许被 config 覆盖的参数
            protected_args = [
                "dataset", "data_path", "exp_index", "results_dir", "model",
                "image_size", "num_classes", "epochs", "global_batch_size", "global_seed",
                "num_workers", "log_every", "vis_every", "ckpt_every", 
                "train_diffusion_steps", "noise_schedule", "mixed_precision", 
                "resume", "disable_compile", "max_grad_norm"
            ]

            # 安全删除 config 中的 protected 参数
            for key in protected_args:
                config_args.pop(key, None)
                
            # 设置默认值
            parser.set_defaults(**config_args)
    
    # re-parse command-line args to overwrite with any command-line inputs
    args = parser.parse_args()
    if args.folder_name is not None:
        args.vq_ckpt = os.path.join(base_path, 'checkpoints', args.checkpoints)
        args.vq_mean, args.vq_std = 0,0
        
    main(args)
