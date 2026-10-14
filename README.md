# MacTok
This repository contains the official implementation of the CVPR 2026 paper **"MacTok: Robust Continuous Tokenization for Image Generation"**(https://cvpr.thecvf.com/virtual/2026/poster/36668).

MacTok is a robust 1D continuous image tokenizer that effectively addresses the **posterior collapse** problem often encountered in extreme compression regimes (e.g., compressing an image to just 64 or 128 1D tokens). By introducing masked representation modeling and global/local representation alignment with DINOv2 features, MacTok achieves state-of-0the-art image generation and reconstruction performance with a remarkably compact latent space.

## Highlights
* **Extreme Compression**: Compresses 256x256 or 512x512 images into only **64 or 128 1D continuous tokens** (up to 64x reduction compared to traditional 2D tokenizers).
* **High-Fidelity Generation**: Achieves state-of-the-art **gFID of 1.52** on ImageNet 512x512 and **1.44** on ImageNet 256x256 when paired with LightningDiT / SiT.
* **Robustness against Posterior Collapse**: Utilizes Random Masking and DINO-guided Semantic Masking to enforce meaningful representation learning even under severe bottlenecks.

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/MacTok.git
cd MacTok
pip install -r requirements.txt
```

*(Optional) Ensure you have `flash_attn` installed for optimized attention computations.*

## Data Preparation

The training scripts expect an ImageNet-style dataset. Organize your dataset directory as follows:

```
ImageNet/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_10026.JPEG
│   │   └── ...
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    └── ...
```
*Note: Update the `data_path` parameter in your `.yaml` configuration files to match your dataset location.*

## Training MacTok Tokenizer

MacTok uses `.yaml` configuration files located in the `configs/tokenizer/` directory. The main configurations are `mactok-b-64.yaml` and `mactok-b-128.yaml`.

To train the tokenizer, use `train/train_tokenizer.py` with `torchrun`:

```bash
torchrun --nproc_per_node=8 train/train_tokenizer.py \
    --config configs/tokenizer/mactok-b-128.yaml
```

## Acknowledgments
This repository is built upon the foundational work of several open-source projects:
* [continuous_tokenizer (SoftVQ-VAE & MAETok)](https://github.com/Hhhhhhao/continuous_tokenizer)
* [LightningDiT (Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models)](https://github.com/hustvl/LightningDiT)
* [SiT (Scalable Interpolant Transformers)](https://github.com/willumn/SiT)
* [DiT (Scalable Diffusion Models with Transformers)](https://github.com/facebookresearch/DiT)
* [timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models)

We thank the authors of these repositories for their excellent work.