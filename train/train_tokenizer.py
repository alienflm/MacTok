# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
from functools import partial
import itertools
import os
import random
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import wandb
import ruamel.yaml as yaml
import numpy as np
from tqdm import tqdm 


import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
import argparse
from glob import glob
from copy import deepcopy

from timm.scheduler import create_scheduler_v2 as create_scheduler

from utils.logger_func import create_logger
from utils.distributed import init_distributed_mode
from utils.misc import str2bool, manage_checkpoints, load_model_state_dict
from utils.optim import param_groups_weight_decay
from utils.data import ImageFolderWithFilename, random_crop_arr, center_crop_arr
from modelling.tokenizer import VQ_models
from losses.vq_loss import VQLoss
from torchsummary import summary

import warnings
warnings.filterwarnings('ignore')
from collections import OrderedDict

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
#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    
    seed = args.global_seed * dist.get_world_size() + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(device)
    

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        if args.exp_index is not None:
            experiment_index = int(args.exp_index)
        else:
            experiment_index = len(glob(f"{args.results_dir}/*")) + 78
        if args.config is not None:
            model_string_name = '.'.join(args.config.split('/')[-1].split('.')[:-1])
            if model_string_name.startswith('exp'):
                model_string_name = '-'.join(model_string_name.split('-')[1:])
        else:
            model_string_name = args.vq_model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/exp{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        experiment_config = vars(args)
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            # Use the round_trip_dump method to preserve the order and style
            file_yaml = yaml.YAML()
            file_yaml.dump(experiment_config, f)
        os.environ["WANDB_MODE"] = "offline"
        wandb_logger = wandb.init(project='tokenizer', name=f'exp{experiment_index:03d}-{model_string_name}')
        # wandb_logger = None
    else:
        logger = create_logger(None)
        wandb_logger = None

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    logger.info(f"RANK={rank}, SEED={seed}, DEVICE={device}")

    # create and load model
    vq_model = VQ_models[args.vq_model](
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
        select_specific_tokens_interval = args.select_specific_tokens_interval,
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
        gamma=args.gamma,
        num_codebooks=args.num_codebooks,
        dec_drop_path_rate=args.dec_drop_path_rate,
        dec_cls_token=args.dec_cls_token,
        # use rope
        use_ape=args.use_ape,
        use_rope=args.use_rope,
        rope_mixed=args.rope_mixed,
        rope_theta=args.rope_theta,
        # encoder mask modeling
        enc_token_drop=args.enc_token_drop,
        # to pixel
        to_pixel=args.to_pixel,
        latent_mask=args.latent_mask,
        enc_dino_mask=args.enc_dino_mask,
    )
    logger.info(f"VQ Model Parameters: {sum(p.numel() for p in vq_model.parameters() if p.requires_grad):,}")
    if args.ema:
        ema = deepcopy(vq_model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"VQ Model EMA Parameters: {sum(p.numel() for p in ema.parameters() if p.requires_grad):,}")
    vq_model = vq_model.to(device)
    # summary(vq_model, input_size=(3, 512, 512))

    vq_loss = VQLoss(
        disc_start=args.disc_start, 
        disc_weight=args.disc_weight,
        disc_type=args.disc_type,
        disc_loss=args.disc_loss,
        disc_dim=args.disc_dim,
        gen_adv_loss=args.gen_loss,
        image_size=args.image_size,
        reconstruction_weight=args.reconstruction_weight,
        specific_rec_loss_weight = args.specific_rec_loss_weight,
        reconstruction_loss=args.reconstruction_loss,
        codebook_weight=args.codebook_weight,
        perceptual_loss=args.perceptual_loss,
        perceptual_model=args.perceptual_model,
        perceptual_dino_variants=args.perceptual_dino_variants,
        perceptual_weight=args.perceptual_weight,
        perceptual_intermediate_loss=args.perceptual_intermediate_loss,
        perceptural_logit_loss=args.perceptual_logit_loss,
        perceptual_resize=args.perceptual_resize,
        perceptual_warmup=args.perceptual_warmup,
        lecam_loss_weight=args.lecam_loss_weight,
        disc_cr_loss_weight=args.disc_cr_loss_weight,
        use_diff_aug=args.use_diff_aug,
        disc_adaptive_weight=args.disc_adaptive_weight,
        repa_adaptive_weight=args.repa_adaptive_weight,
        wandb_logger=wandb_logger,
        use_disc = args.use_disc
    ).to(device)
    logger.info(f"Discriminator Parameters: {sum(p.numel() for p in vq_loss.discriminator.parameters() if p.requires_grad):,}")
    
    # scaling lr
    args.lr = args.lr * args.global_batch_size / 256
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    scaler_disc = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    # Setup optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(vq_model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        optimizer_disc = torch.optim.Adam(vq_loss.discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(param_groups_weight_decay(vq_model, weight_decay=args.weight_decay), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        optimizer_disc = torch.optim.AdamW(param_groups_weight_decay(vq_loss.discriminator, weight_decay=args.weight_decay), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
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

    num_update_steps_per_epoch = len(loader)
    max_train_steps = args.epochs * num_update_steps_per_epoch

    # create lr scheduler
    if args.lr_scheduler == 'none':
        vqvae_lr_scheduler = None
        disc_lr_scheduler = None
    else:
        vqvae_lr_scheduler, _ = create_scheduler(
            sched=args.lr_scheduler,
            optimizer=optimizer,
            patience_epochs=0,
            step_on_epochs=False,
            updates_per_epoch=num_update_steps_per_epoch,
            num_epochs=args.epochs,
            warmup_epochs=args.lr_warmup_epochs,
            min_lr=args.lr * 0.1,
        )
        disc_lr_scheduler, _ = create_scheduler(
            sched=args.lr_scheduler,
            optimizer=optimizer_disc,
            patience_epochs=0,
            step_on_epochs=False,
            updates_per_epoch=num_update_steps_per_epoch,
            num_epochs=args.epochs,
            warmup_epochs=args.lr_warmup_epochs,
            min_lr=args.lr * 0.1,
        )

    logger.info(f"num_update_steps_per_epoch {num_update_steps_per_epoch:,} max_train_steps ({max_train_steps})")

    # Prepare models for training:
    if args.vq_ckpt:
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        
        vq_model.load_state_dict(load_model_state_dict(checkpoint['model']))
        if args.ema:
            ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        vq_loss.discriminator.load_state_dict(load_model_state_dict(checkpoint["discriminator"]))
        optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
        if not args.finetune:
            train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.vq_ckpt.split('/')[-1].split('.')[0])
            start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
            train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))
        else:
            train_steps = 0
            start_epoch = 0           
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.vq_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
    
    if args.compile:
        logger.info("compiling the model... (may take several minutes)")
        vq_model = torch.compile(vq_model) # requires PyTorch 2.0      
        vq_loss.discriminator = torch.compile(vq_loss.discriminator)  
        vq_loss.perceptual_loss = torch.compile(vq_loss.perceptual_loss)
        logger.info("compiling done.")
    
    # vq_model = DDP(vq_model.to(device), device_ids=[args.gpu], find_unused_parameters=True) # checkpoint
    vq_model = DDP(vq_model, device_ids=[args.gpu], find_unused_parameters=True) # checkpoint
    if args.ema:
        update_ema(ema,vq_model.module._orig_mod if args.compile else vq_model.module, decay=0)  # Ensure EMA is initialized with synced weights    
    vq_model.train()
    
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode
    vq_loss = DDP(vq_loss.to(device), device_ids=[args.gpu])
    vq_loss.train()

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    is_main_process = not dist.is_initialized() or dist.get_rank() == 0
    pbar = tqdm(range(start_epoch, args.epochs), desc="Training epochs") if is_main_process else range(start_epoch, args.epochs)
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in pbar:
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
                
            imgs = x.to(device, non_blocking=True)

            # generator training
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):  
                recons_imgs, codebook_loss, info = vq_model(imgs)

                loss_gen = vq_loss(codebook_loss, imgs, recons_imgs, optimizer_idx=0, global_step=train_steps+1, 
                                   last_layer=vq_model.module.decoder.last_layer,
                                   logger=logger, log_every=args.log_every,
                                   encoder_last_layer = vq_model.module.encoder.last_layer)
            scaler.scale(loss_gen).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(vq_model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if args.ema:
                if args.ema_decay > 0.0:
                    update_ema(ema, vq_model.module._orig_mod if args.compile else vq_model.module,decay=args.ema_decay)
                else:
                    update_ema(ema, vq_model.module._orig_mod if args.compile else vq_model.module)

            # discriminator training            
            optimizer_disc.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):
                loss_disc = vq_loss(codebook_loss, imgs, recons_imgs, optimizer_idx=1, global_step=train_steps+1,
                                    logger=logger, log_every=args.log_every)
            scaler_disc.scale(loss_disc).backward()
            if args.max_grad_norm != 0.0:
                scaler_disc.unscale_(optimizer_disc)
                torch.nn.utils.clip_grad_norm_(vq_loss.module.discriminator.parameters(), args.max_grad_norm)
            scaler_disc.step(optimizer_disc)
            scaler_disc.update()
            
            # # Log loss values:
            running_loss += loss_gen.item() + loss_disc.item()
            
            
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}/total_steps:{max_train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()
            
                if rank == 0 and wandb_logger is not None:
                    log_dict = {"lr": optimizer.param_groups[0]["lr"], "train_loss": avg_loss}
                    wandb_logger.log(log_dict,
                        step=train_steps
                    )
                
            if train_steps % args.vis_every == 0:
                if not isinstance(recons_imgs, tuple):
                    image = torch.cat([imgs[:4], recons_imgs[:4]], dim=0)
                    image = torch.clamp(image, min=-1, max=1)
                    image = make_grid((image + 1) / 2, nrow=4, padding=0, pad_value=1.0)
                    image = image.permute(1, 2, 0).mul_(255).cpu().numpy()
                    image = Image.fromarray(image.astype(np.uint8))
                else:
                    image = torch.cat([imgs[:4], recons_imgs[0][:4],recons_imgs[1][:4]], dim=0)
                    image = torch.clamp(image, min=-1, max=1)
                    image = make_grid((image + 1) / 2, nrow=4, padding=0, pad_value=1.0)
                    image = image.permute(1, 2, 0).mul_(255).cpu().numpy()
                    image = Image.fromarray(image.astype(np.uint8))                    

                if rank == 0:
                    wandb_logger.log({"recon_images": [wandb.Image(image)]}, step=train_steps)

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:

                if rank == 0:
                    if args.compile:
                        model_weight = vq_model.module._orig_mod.state_dict()
                    else:
                        model_weight = vq_model.module.state_dict()  
                    if args.opt_save:
                        checkpoint = {
                            "model": model_weight,
                            "optimizer": optimizer.state_dict(),
                            "discriminator": vq_loss.module.discriminator.state_dict(),
                            "optimizer_disc": optimizer_disc.state_dict(),
                            "steps": train_steps,
                            "args": args
                        }
                    else:
                        checkpoint = {
                            "model": model_weight,
                            "discriminator": vq_loss.module.discriminator.state_dict(),
                            "steps": train_steps,
                            "args": args
                        }                        
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    # if not args.no_local_save:
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    manage_checkpoints(checkpoint_dir,args.keep_last_n)
                dist.barrier()

            if vqvae_lr_scheduler is not None:
                vqvae_lr_scheduler.step_update(train_steps)
            if disc_lr_scheduler is not None:
                disc_lr_scheduler.step_update(train_steps)


    vq_model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/tokenizer/cnn_llamagen_vq16.yaml', help="config file used to specify parameters")
    
    parser.add_argument("--opt-save", type=str2bool, default=True, help="opt save")
    parser.add_argument("--keep-last-n", type=int, default=20, help="keep last n checkpoints")
    parser.add_argument("--exp-index", type=str, default=None, help="experiment index")
    parser.add_argument("--data-path", type=str, default="ImageNet2012/train")
    parser.add_argument("--cloud-save-path", type=str, help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", type=str2bool, default=False, help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--finetune", type=str2bool, default=False, help="finetune a pre-trained vq model")
    parser.add_argument("--ema", type=str2bool, default=True, help="whether using ema training")
    parser.add_argument("--ema-decay", type=float, default=0.0, help="ema decay")
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
    
    parser.add_argument("--use-disc", type=str2bool, default=True, help="use discriminator for gan training")
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
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
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
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--vis-every", type=int, default=5000)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 

    parser.add_argument("--enc-type", type=str, default="cnn")
    parser.add_argument("--dec-type", type=str, default="cnn")
    parser.add_argument("--num-latent-tokens", type=int, default=None)
    parser.add_argument("--specific-rec-loss-weight", type=float, default=0.1, help="specific reconstruction loss weight of image pixel")
    parser.add_argument("--num-generic-latent-tokens", type=int, default=0, help="number of generic latent tokens")
    parser.add_argument("--num-specific-latent-tokens", type=int, default=0, help="number of specific latent tokens")
    parser.add_argument("--specific-recon", type=bool, default=False, help="whether specific reconstruction")
    parser.add_argument("--select-specific-tokens-interval", type=int, default=0, help="whether select specific tokens")
    parser.add_argument("--encoder-model", type=str, default='vit_small_patch14_dinov2.lvd142m', help='encoder model name')
    parser.add_argument("--decoder-model", type=str, default='vit_small_patch14_dinov2.lvd142m', help='decoder model name')
    parser.add_argument("--encoder-tuning-method", type=str, default='full', help='tuning method for encoder', choices=['full', 'lora', 'frozen'])
    parser.add_argument("--decoder-tuning-method", type=str, default='full', help='tuning method for decoder', choices=['full', 'lora', 'frozen'])
    parser.add_argument("--encoder-pretrained", type=str2bool, default=True, help='load pre-trained weight for encoder')
    parser.add_argument("--decoder-pretrained", type=str2bool, default=False, help='load pre-trained weight for decoder')
    parser.add_argument("--encoder-patch-size", type=int, default=16, help='encoder patch size')
    parser.add_argument("--decoder-patch-size", type=int, default=16, help='decoder patch size')
    parser.add_argument("--dec-drop-path-rate", type=float, default=0.0, help='decoder drop path rate')
    parser.add_argument("--dec-cls-token", type=str2bool, default=True, help='decoder cls token')
    parser.add_argument("--to-pixel", type=str, default="linear")
    
    # repa
    parser.add_argument("--repa", type=str2bool, default=False, help='use repa')
    # parser.add_argument("--repa-stop-steps", type=int, default=-1, help='repa stop steps')
    parser.add_argument('--repa-model', type=str, default='vit_base_patch16_224', help='repa model name')
    parser.add_argument('--repa-patch-size', type=int, default=16, help='repa patch size')
    parser.add_argument('--repa-proj-dim', type=int, default=1024, help='repa embed dim')
    parser.add_argument('--repa-loss-weight', type=float, default=0.1, help='repa loss weight')
    parser.add_argument('--repa-cls-weight', type=float, default=0.1, help='repa cls loss weight')
    parser.add_argument('--repa-align', type=str, default='global', help='align repa feature', choices=['global', 'avg_1d', 'avg_2d', 'avg_1d_shuffle'])
    parser.add_argument('--repa-cls', type=str2bool, default=False, help='repa specific token to cls token')
    parser.add_argument('--repa-adaptive-weight', type=str2bool, default=False, help='repa_adaptive_weight')
    
    
    # noise scale
    parser.add_argument("--gamma", type=float, default=0.0, help='noise level')
    parser.add_argument("--noise-type", type=str, default='linear', help='noise type')
    
    
    # mask modeling
    # make sure drop is 0.0 for not using mask modeling
    parser.add_argument("--enc-token-drop", type=float, default=0.0, help='encoder token drop')
    parser.add_argument("--enc-token-drop-max", type=float, default=0.75, help='maximum drop rate')

    parser.add_argument("--use-ape", type=str2bool, default=True, help='ape poistional embedding')  
    parser.add_argument("--use-rope", type=str2bool, default=False, help='rope poistional embedding')  
    parser.add_argument("--rope-mixed", type=str2bool, default=False, help='rope poistional embedding')  
    parser.add_argument("--rope-theta", type=float, default=10.0, help='rope poistional embedding')  
    
    # latent mask
    parser.add_argument("--latent-mask", type=float, default=0.0, help='latent mask ratio')
    parser.add_argument("--latent-mask-mixed", type=bool, default=False, help='GPU id for training')
    
    
    parser.add_argument("--enc-dino-mask", type=bool, default=False, help='whether to use mixed mask strategy')
    parser.add_argument("--const-mask", type=str2bool, default=False, help='constant mask during training') 
    parser.add_argument("--mask-model", type=str, default=None, help='mask model')
    parser.add_argument("--take-indices", type=int, default=None, help='whether to use dino mask for encoder')
    
    #fFirst parse of command-line args to check for config file
    args = parser.parse_args()
    
    # If a config file is specified, load it and set defaults
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**config_args)
    
    # re-parse command-line args to overwrite with any command-line inputs
    args = parser.parse_args()
    main(args)
