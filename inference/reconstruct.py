import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm
import os
import sys
from PIL import Image
import numpy as np
import random
import argparse
import itertools
from utils.misc import str2bool,load_model_state_dict
import ruamel.yaml as yaml
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from modelling.tokenizer import VQ_models

from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

from utils.data import center_crop_arr
from modelling.tokenizer import SoftVQModel
from torchmetrics.image.fid import FrechetInceptionDistance

def create_npz_from_sample_folder(sample_dir, num=50000):
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
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path



def main(args):
    
    
    # Setup PyTorch:
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    if args.no_distributed:
        rank = 0
        world_size = 1
    else:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    
    # vq_model = SoftVQModel.from_pretrained(args.vq_model)
    # vq_model.eval()
    
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
        enc_tuning_method=args.encoder_tuning_method,
        dec_tuning_method=args.decoder_tuning_method,
        enc_pretrained=args.encoder_pretrained,
        dec_pretrained=args.decoder_pretrained,
        enc_patch_size=args.encoder_patch_size,
        dec_patch_size=args.decoder_patch_size,
        dec_drop_path_rate=args.dec_drop_path_rate,
        dec_cls_token=args.dec_cls_token,
        tau=args.tau,
        repa=args.repa,
        repa_model=args.repa_model,
        repa_patch_size=args.repa_patch_size,
        repa_proj_dim=args.repa_proj_dim,
        repa_loss_weight=args.repa_loss_weight,
        repa_align=args.repa_align,
        num_codebooks=args.num_codebooks,
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
        enc_dino_mask=args.enc_dino_mask
    )
    if args.vq_ckpt:
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        vq_model.load_state_dict(load_model_state_dict(checkpoint['ema']),strict=False)
    # vq_model, _ , vae_embed_dim, dit_input_size, latents_bias, latents_scale, vae_1d,config = build_tokenizer(args.config, args.vq_ckpt)
    vq_model.eval()
    vq_model = vq_model.to(device)
        
    # Create folder to save samples:
    # folder_name = (f"{args.vq_model}-{args.dataset}-size-{args.image_size}-size-{args.image_size_eval}-seed-{args.global_seed}")
    folder_name = args.folder_name
    sample_folder_dir = f"{args.sample_dir}/{folder_name}_{args.checkpoints[:-3]}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = ImageFolder(args.data_path, transform=transform)
    num_fid_samples = 9600
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=args.per_proc_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )    

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * world_size
    
    psnr_val_rgb = []
    ssim_val_rgb = []
    compute_fid_score = FrechetInceptionDistance(normalize=False).cuda()
    all_z = []
    all_zq = []
    loader = tqdm(loader) if rank == 0 else loader
    total = 0
    for x, _ in loader:
        if args.image_size_eval != args.image_size:
            rgb_gts = F.interpolate(x, size=(args.image_size_eval, args.image_size_eval), mode='bicubic')
        else:
            rgb_gts = x
        rgb_gts = (rgb_gts.permute(0, 2, 3, 1).to("cpu").numpy() + 1.0) / 2.0 # rgb_gt value is between [0, 1]
        x = x.to(device, non_blocking=True)
        with torch.no_grad():
            
            samples = vq_model(x)

            if args.image_size_eval != args.image_size:
                samples = F.interpolate(samples, size=(args.image_size_eval, args.image_size_eval), mode='bicubic')
        
        
        samples = torch.clamp(127.5 * samples + 128, 0, 255).to(dtype=torch.uint8)
        x = torch.clamp(127.5 * x + 128, 0, 255).to(dtype=torch.uint8)
    
        compute_fid_score.update(x, real=True)
        compute_fid_score.update(samples, real=False)
        
        samples = samples.permute(0, 2, 3, 1).to("cpu").numpy()

        # Save samples to disk as individual .png files
        for i, (sample, rgb_gt) in enumerate(zip(samples, rgb_gts)):
            index = i * world_size + rank + total
            # if i==0:
            #     Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
            # metric
            rgb_restored = sample.astype(np.float32) / 255. # rgb_restored value is between [0, 1]
            psnr = psnr_loss(rgb_restored, rgb_gt)
            ssim = ssim_loss(rgb_restored, rgb_gt, multichannel=True, data_range=2.0, channel_axis=-1)
            psnr_val_rgb.append(psnr)
            ssim_val_rgb.append(ssim)
            
        total += global_batch_size

    if rank == 0:
        print('\n*************开始计算FID指标**************\n')        
    fid = compute_fid_score.compute().detach()
    
    # ------------------------------------
    #       Summary
    # ------------------------------------
    # Make sure all processes have finished saving their samples
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        # all_gather_object 需要传入长度为world_size的list作为容器
        gather_psnr_val = [[] for _ in range(world_size)]
        gather_ssim_val = [[] for _ in range(world_size)]
        
        # all_gather_object 会把各个进程的对象放入对应位置
        dist.all_gather_object(gather_psnr_val, psnr_val_rgb)
        dist.all_gather_object(gather_ssim_val, ssim_val_rgb)
    else:
        # 非分布式时，world_size和rank默认值
        gather_psnr_val = [psnr_val_rgb]
        gather_ssim_val = [ssim_val_rgb]
    

    # print(gather_latents)
    if rank == 0:
    # all_gather_object 会返回嵌套列表，非分布式时无需拆包
        gather_psnr_val = list(itertools.chain(*gather_psnr_val))
        gather_ssim_val = list(itertools.chain(*gather_ssim_val))
        # gather_fid_val = list(itertools.chain(*gather_fid_val))     
        psnr_val_rgb = sum(gather_psnr_val) / len(gather_psnr_val)
        ssim_val_rgb = sum(gather_ssim_val) / len(gather_ssim_val)
        # fid_val = sum(gather_fid_val) / len(gather_fid_val)
        
        print("PSNR: %f, SSIM: %f " % (psnr_val_rgb, ssim_val_rgb))
        print("FID: %f" % (fid))
        
        result_file = f"{sample_folder_dir}_results.txt"
        print("writing results to {}".format(result_file))
        with open(result_file, 'w') as f:
            print("PSNR: %f, SSIM: %f " % (psnr_val_rgb, ssim_val_rgb), file=f)
            print("FID: %f " % (fid), file=f)
            
        # create_npz_from_sample_folder(sample_folder_dir, num_fid_samples)
        # print("Done.")

    
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=['imagenet', 'coco'], default='imagenet')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--image-size-eval", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--sample-dir", type=str, default="reconstructions")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-distributed", type=str2bool, default=False)

    parser.add_argument("--config", type=str, default='configs/tokenizer/cnn_llamagen_vq16.yaml', help="config file used to specify parameters")
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    # parser.add_argument("--vq-model", type=str, default="SoftVQVAE/softvq-l-64")
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
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--vis-every", type=int, default=5000)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 

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
    parser.add_argument("--dec-drop-path-rate", type=float, default=0.0, help='decoder drop path rate')
    parser.add_argument("--dec-cls-token", type=str2bool, default=True, help='decoder cls token')    
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

    parser.add_argument("--use-ape", type=str2bool, default=True, help='ape poistional embedding')  
    parser.add_argument("--use-rope", type=str2bool, default=False, help='rope poistional embedding')  
    parser.add_argument("--rope-mixed", type=str2bool, default=False, help='rope poistional embedding')  
    parser.add_argument("--rope-theta", type=float, default=10.0, help='rope poistional embedding')     
    
    parser.add_argument('--folder-name', type=str, default=None, help='Name of the experiment folder')
    parser.add_argument('--checkpoints', type=str, default='0010000.pt', help='0010000.pt')
    
    # latent mask
    parser.add_argument("--latent-mask", type=float, default=0.0, help='latent mask ratio')
    parser.add_argument("--enc-dino-mask", type=bool, default=False, help='whether to use mixed mask strategy')
        
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
    if args.folder_name is not None:
        args.vq_ckpt = os.path.join(base_path, 'checkpoints', args.checkpoints)
        
    main(args)
    
