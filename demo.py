import os
from argparse import Namespace
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import datetime
import time
import math
from pathlib import Path
from tqdm import tqdm
import torch.distributed as dist
from contextlib import nullcontext
from datasets.nerf_dataset import get_nerf_dataloader
from datasets.online_process_dataset import get_online_process_dataloader, SpikeConverter
from models.spike_mar import spike_mar_base, spike_mar_large, spike_mar_huge
# from models.vae import AutoencoderKL
from diffusers import AutoencoderKL
from util import misc
from util.metrics import compute_img_metric
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.misc import AverageMeter
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

import os

os.environ["LOCAL_RANK"] = "0"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

def load_model(args):
    """Load all model components"""
    print("Loading model...")
    vae2d = AutoencoderKL.from_pretrained(args.rgb_vae_path).eval()
    
    model_func = globals()[args.model]
    model = model_func(
        img_h=args.img_h,
        img_w=args.img_w,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.rgb_vae_embed_dim,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        buffer_size=args.buffer_size,
        diffloss_d=args.diffloss_d,
        diffloss_w=args.diffloss_w,
        num_sampling_steps=args.num_sampling_steps,
        diffusion_batch_mul=args.diffusion_batch_mul,
        grad_checkpointing=args.grad_checkpointing,
    )

    
    # Load Diffusion model checkpoint (if provided)
    if args.diffusion_ckpt and os.path.isfile(args.diffusion_ckpt):
        print(f"Loading Diffusion model checkpoint from {args.diffusion_ckpt}")
        checkpoint = torch.load(args.diffusion_ckpt, map_location='cpu', weights_only=False)
        
        # Get model state dict (use model_ema if exists)
        if 'model_ema' in checkpoint:
            ckpt_state_dict = checkpoint['model_ema']
        else:
            ckpt_state_dict = checkpoint['model']
        
        pos_embed_keys = [
            'encoder_pos_embed_learned',
            'decoder_pos_embed_learned',
            'diffusion_pos_embed_learned',
        ]
        
        # Create new state dict excluding all keys to ignore
        ckpt_state_dict = {k: v for k, v in ckpt_state_dict.items() 
                               if not any(pos_key in k for pos_key in pos_embed_keys)}
        
        # Load with strict=False to allow mismatch
        missing_keys, unexpected_keys = model.load_state_dict(ckpt_state_dict, strict=False)
        
        print(f"Loaded successfully!")
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        
    for param in vae2d.parameters():
        param.requires_grad = False
    # Print the trainable parameters and relative ratio
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params}, Total params: {total_params}, Trainable ratio: {trainable_params/total_params:.2%}")
    
    return model, vae2d


def load_data(args):
    """Load dataset"""
    print("Loading dataset...")
    train_loader = get_online_process_dataloader(
        image_folder=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_h=args.img_h,
        img_w=args.img_w,
        shuffle=True,
        distributed=args.distributed,
        rank=args.rank,
        world_size=args.world_size,
        # Image processing parameters
        kernel_size=args.kernel_size,
        blur_intensity=args.blur_intensity,
        blur_samples=args.blur_samples
    )
    
    dataset_size = len(train_loader.dataset)
    if dataset_size == 0:
        raise ValueError(f"Dataset is empty, please check path: {args.data_root}")
    
    print(f"Dataset loaded, total samples: {dataset_size}")
    
    # Create spike converter
    spike_converter = SpikeConverter(
        photon_samples=args.photon_samples,
        target_coverage=args.target_coverage,
        smooth_sigma=args.smooth_sigma,
        gamma=args.gamma
    )
    
    # Get one sample to set frame resolution
    data = next(iter(train_loader))
    # Generate spike data from RGB
    spike_data = spike_converter.rgb_to_spike(data['rgb'])
    args.frame_resolution = spike_data.shape[1]
    
    # Print data shapes for verification
    print(f"Shape check:")
    print(f"- RGB image: {data['rgb'].shape}")
    print(f"- Blur image: {data['blur'].shape}")
    print(f"- Spike data: {spike_data.shape}")
    
    return train_loader, spike_converter
def unwrap_model(model, distributed=True):
    """Unwrap model from DistributedDataParallel"""
    if distributed and hasattr(model, 'module'):
        return model.module
    return model


def compute_metric(args, rgb_gt, generated_images):
    # calculate the metric
    metrics = {}
    method_list = ['mar']
    metric_list = ['mse','ssim','psnr','lpips']
    for method_name in method_list:
        metrics[method_name] = {}
        for metric_name in metric_list:
            metrics[method_name][metric_name] = AverageMeter()
    for key in metric_list :
        metrics['mar'][key].update(compute_img_metric(rgb_gt, generated_images, key))
    output_path = args.output_dir + "/metrics.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    for method_name in method_list:
        re_msg = ''
        for metric_name in metric_list:
            re_msg += metric_name + ": " + "{:.3f}".format(metrics[method_name][metric_name].avg) + "  "
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(f"{method_name}: {re_msg}" + '\n')
        print(f"{method_name}: {re_msg}")


def evaluate(model, vae2d, viz_data, device, epoch, step, log_writer, args, spike_converter=None):
    """Evaluate the model and generate visualization results"""
    if not misc.is_main_process():
        return
        
    model.eval()
    model_unwrapped = model
    
    # Get samples from the dataset for visualization
    blur_images = viz_data['blur'].to(device)[:8]  # Use only the first 8 samples
    rgb_gt = viz_data['rgb'].to(device)[:8]
    
    # Generate spike data directly on GPU
    spike_stream = spike_converter.rgb_to_spike(rgb_gt)[:8]
    
    # Directory to save output images
    output_dir = Path(args.output_dir) / 'generations'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get latent representations via VAE encoder
    with torch.no_grad():
        # Encode blurred RGB
        rgb_blur_latent = vae2d.encode(blur_images).latent_dist.sample().mul_(args.vae_scale)
        vae_recon_blur = vae2d.decode(rgb_blur_latent / args.vae_scale)[0]
        rgb_gt_latent = vae2d.encode(rgb_gt).latent_dist.sample().mul_(args.vae_scale)
        vae_recon_gt = vae2d.decode(rgb_gt_latent / args.vae_scale)[0]
        
        # Generate samples
        generated_tokens = model_unwrapped.recon_tokens(
            blur_latent = rgb_blur_latent, 
            spike_stream = spike_stream,
            temperature=0.5
        )
        
        no_spike_generated_tokens = model_unwrapped.recon_tokens(
            blur_latent = rgb_blur_latent, 
            spike_stream=None,
            temperature=0.5
        )
        
        no_rgb_generated_tokens = model_unwrapped.recon_tokens(
            blur_latent = None, 
            spike_stream = spike_stream,
            temperature=0.5
        )
        
        # Decode generated latent representations
        generated_images = vae2d.decode(generated_tokens / args.vae_scale)[0]
        no_spike_generated_images = vae2d.decode(no_spike_generated_tokens / args.vae_scale)[0]
        no_rgb_generated_images = vae2d.decode(no_rgb_generated_tokens / args.vae_scale)[0]
        
        # Normalize to valid range
        blur_images = (blur_images + 1) / 2
        rgb_gt = (rgb_gt + 1) / 2
        generated_images = (generated_images + 1) / 2
        no_spike_generated_images = (no_spike_generated_images + 1) / 2
        no_rgb_generated_images = (no_rgb_generated_images + 1) / 2
        vae_recon_blur = (vae_recon_blur + 1) / 2
        vae_recon_gt = (vae_recon_gt + 1) / 2
        
        # pick a spike frame from spike stream
        spike_frame = rgb_gt*spike_stream[:, :3, :, :]
        compute_metric(args, rgb_gt, generated_images)

    # Create summary comparison image
    comparison = torch.cat([
        blur_images, 
        spike_frame,
        vae_recon_blur,
        no_spike_generated_images,
        no_rgb_generated_images,
        generated_images,
        vae_recon_gt,
        rgb_gt,
        ], dim=0)
    grid = make_grid(comparison, nrow=8, normalize=True, padding=2)
    
    if log_writer is not None:
        log_writer.add_image('Generated/comparison', grid, epoch)
    
    # Save image using torchvision's save_image instead of matplotlib
    save_image(grid, str(output_dir / f'comparison_epoch_{epoch}_{step}.png'))
    
    # Resume training mode
    model.train()



class SpikeGenArgs:
    def __init__(self):
        # Basic training params
        self.batch_size = 64
        self.epochs = 100
        self.model = 'spike_mar_base'

        # VAE parameters
        self.img_h = 256
        self.img_w = 256
        self.rgb_vae_path = "ostris/vae-kl-f8-d16"
        self.rgb_vae_embed_dim = 16
        self.vae_stride = 8
        self.patch_size = 1
        self.vae_scale = 0.2325

        # Generation parameters
        self.eval_freq = 100
        self.save_last_freq = 10
        self.online_eval = False

        # Optimizer parameters
        self.weight_decay = 0.02
        self.grad_checkpointing = False
        self.lr = None
        self.blr = 1e-4
        self.min_lr = 0.0
        self.lr_schedule = 'cosine'
        self.warmup_epochs = 10

        # MAR parameters
        self.diffusion_ckpt = '/path/to/your/diffusion_checkpoint.pth'
        self.grad_clip = 3.0
        self.attn_dropout = 0.1
        self.proj_dropout = 0.1
        self.buffer_size = 64

        # Diffusion loss parameters
        self.diffloss_d = 6
        self.diffloss_w = 1024
        self.num_sampling_steps = "100"
        self.diffusion_batch_mul = 1
        self.temperature = 1.0

        # Dataset parameters
        self.data_root = '/path/to/your/dataset'
        self.scenes = None  # Expect list[str] if used
        self.output_dir = './output/spikegen_pretrained'
        self.log_dir = './output_dir'
        self.device = 'cuda'
        self.seed = 0
        self.resume = ''

        self.start_epoch = 0
        self.num_workers = 10
        self.pin_mem = True

        # Distributed training
        self.world_size = 1
        self.local_rank = -1
        self.dist_on_itp = False
        self.dist_url = 'env://'

        # Image processing
        self.kernel_size = 40
        self.blur_intensity = 40.0
        self.blur_samples = 8
        self.photon_samples = 8
        self.target_coverage = 0.1
        self.smooth_sigma = 1.0
        self.gamma = 2.0  # gamma correction




args = SpikeGenArgs()
# Initialize distributed training
misc.init_distributed_mode(args)


device = torch.device(args.device)

# Fix random seed
seed = args.seed + misc.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
cudnn.benchmark = True


# Set data loader
train_loader, spike_converter = load_data(args)



# Load model
model, vae2d = load_model(args)
# Move to target device
model.to(device)
vae2d.to(device)

torch.cuda.empty_cache()  # Clear cache to avoid OOM
viz_data = next(iter(train_loader))

evaluate(model, vae2d, viz_data, device, 0, -1, None, args, spike_converter)
torch.cuda.empty_cache()


