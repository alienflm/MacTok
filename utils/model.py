from pathlib import Path
import yaml

import torch
from modelling.tokenizer import VQ_models



def build_tokenizer(vq_config,
                    vq_ckpt,latent_norm=False):
    
    with open(vq_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config_name = vq_config.split('/')[-2]
    
    # Every config key that may be forwarded to the tokenizer constructor.
    possible_args = [
        'image_size', 'codebook_size', 'codebook_embed_dim', 'codebook_l2_norm',
        'commit_loss_beta', 'entropy_loss_ratio', 'vq_loss_ratio', 'kl_loss_weight',
        'dropout_p', 'enc_type', 'encoder_model', 'dec_type', 'decoder_model',
        'num_latent_tokens', 'encoder_tuning_method', 'decoder_tuning_method', 
        'encoder_pretrained', 'decoder_pretrained', 'encoder_patch_size', 'decoder_patch_size', 
        'tau', 'repa', 'repa_model', 'repa_patch_size', 'repa_proj_dim', 'repa_loss_weight',
        'repa_align', 'num_codebooks', 'enc_token_drop', 'to_pixel',
        'use_ape','use_rope','rope_mixed','rope_theta','dec_cls_token',
    ]

    # Assemble the keyword arguments, forwarding only keys present in the config.
    params = {}
    for arg in possible_args:
        value = config.get(arg)
        if value is not None:
            if arg.startswith('encoder_') and arg not in ['encoder_model']:
                new_key = arg.replace('encoder_', 'enc_')
            elif arg.startswith('decoder_') and arg not in ['decoder_model']:
                new_key = arg.replace('decoder_', 'dec_')
            else:
                new_key = arg
            params[new_key] = value

    vae = VQ_models[config['vq_model']](**params)
    vae_embed_dim = config['codebook_embed_dim']

    # vq_model.to(device)
    # vq_model.eval()
    checkpoint = torch.load(vq_ckpt, map_location="cpu")
    if "ema" in checkpoint:  # ema
        model_weight = checkpoint["ema"]
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    missing_keys, unexpected_keys = vae.load_state_dict(model_weight, strict=False)
    if missing_keys:
        print(f"[tokenizer] missing keys in checkpoint: {len(missing_keys)}")
    if unexpected_keys:
        print(f"[tokenizer] unexpected keys in checkpoint: {len(unexpected_keys)}")

    vq_1d = False
    if latent_norm:
        import os
        vq_name = vq_ckpt.split('/')[-3]
        ckpt_iter = vq_ckpt.split('/')[-1].split('.')[0]
        latent_stats_cache_file = os.path.join(f'./reconstructions/{vq_name}_{ckpt_iter}',"latents_stats.pt")
        latent_stats = torch.load(latent_stats_cache_file)
        vq_mean, vq_std = latent_stats['mean'], latent_stats['std']
        dit_input_size = config['num_latent_tokens']
        vq_1d = True
    else:
        raise NotImplementedError
    
    
    return vae, config['vq_model'],vae_embed_dim, dit_input_size, vq_mean, vq_std, vq_1d,config