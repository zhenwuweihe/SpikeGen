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
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from codes.utils import * 
from codes.dataset import *
from codes.metrics import compute_img_metric

torch.serialization.add_safe_globals([Namespace])

def get_args_parser():
    parser = argparse.ArgumentParser('Spike-MAR Training', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int,
                      help='Batch size per GPU')
    parser.add_argument('--epochs', default=100, type=int)
    
    # Model parameters
    parser.add_argument('--model', default='spike_mar_large', type=str, metavar='MODEL',
                      help='Name of model to train')
    
    # VAE parameters
    parser.add_argument('--img_h', default=360, type=int,
                      help='Input image height')
    parser.add_argument('--img_w', default=640, type=int,
                      help='Input image width')
    parser.add_argument('--rgb_vae_path', default="pretrained_models/vae/kl16.ckpt", type=str,
                      help='Pretrained RGB VAE path')
    parser.add_argument('--rgb_vae_embed_dim', default=16, type=int,
                      help='RGB VAE embedding dimension')
    parser.add_argument('--vae_stride', default=8, type=int,
                      help='Tokenizer stride')
    parser.add_argument('--patch_size', default=1, type=int,
                      help='Number of tokens grouped as a patch')
    parser.add_argument('--vae_scale', default=0.2325, type=float,
                      help='VAE scaling factor')
    
    # Generation parameters
    parser.add_argument('--num_iter', default=64, type=int,
                      help='Autoregressive iteration count for image generation')
    parser.add_argument('--condition_drop_prob', default=0.1, type=float,
                      help='Condition embedding drop probability')
    parser.add_argument('--eval_freq', type=int, default=1,
                      help='Evaluation frequency')
    parser.add_argument('--save_last_freq', type=int, default=5,
                      help='Save last checkpoint frequency')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--eval_bsz', type=int, default=8,
                      help='Evaluation batch size')
    
    # MAR parameters
    parser.add_argument('--diffusion_ckpt', default='pretrained_models/mar/mar_base/checkpoint-last.pth', type=str,
                    help='Path to diffusion model checkpoint')
    parser.add_argument('--mask_ratio_min', type=float, default=0.7,
                      help='Minimum mask ratio')
    parser.add_argument('--grad_clip', type=float, default=3.0,
                      help='Gradient clipping')
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                      help='Attention dropout')
    parser.add_argument('--proj_dropout', type=float, default=0.1,
                      help='Projection dropout')
    parser.add_argument('--buffer_size', type=int, default=64)
    
    # Diffusion Loss parameters
    parser.add_argument('--diffloss_d', type=int, default=6)
    parser.add_argument('--diffloss_w', type=int, default=1024)
    parser.add_argument('--num_sampling_steps', type=str, default="100")
    parser.add_argument('--diffusion_batch_mul', type=int, default=1)
    parser.add_argument('--temperature', default=1.0, type=float,
                      help='Sampling temperature for diffusion loss')
    
    # Dataset parameters
    parser.add_argument('--data_root', default='./datasets/images',
                      help='Path to image folder')
    parser.add_argument('--scenes', type=str, nargs='+', default=None,
                      help='List of scene names')
    
    parser.add_argument('--output_dir', default='./output_dir',
                      help='Output path')
    parser.add_argument('--log_dir', default='./output_dir',
                      help='Path for tensorboard logs')
    parser.add_argument('--device', default='cuda',
                      help='Device')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                      help='Resume from checkpoint')
    
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                      help='Start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                      help='Pin memory')
    parser.set_defaults(pin_mem=True)
    
    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                      help='Number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                      help='URL used to set up distributed training')
    
    # Image processing related parameters
    parser.add_argument('--kernel_size', type=int, default=50,
                      help='Blur kernel size')
    parser.add_argument('--blur_intensity', type=float, default=50,
                      help='Blur intensity')
    parser.add_argument('--blur_samples', type=int, default=10,
                      help='Number of blur samples')
    parser.add_argument('--photon_samples', type=int, default=8,
                      help='Number of photon samples')
    parser.add_argument('--target_coverage', type=float, default=0.1,
                      help='Target sampling coverage')
    parser.add_argument('--smooth_sigma', type=float, default=1,
                      help='Smoothing sigma')
    parser.add_argument('--gamma', type=float, default=1,
                      help='Gamma correction')
    
    return parser

def load_model(args):
    """Load all model components"""
    print("Loading model...")
    
    # vae2d = AutoencoderKL(
    #     embed_dim=args.rgb_vae_embed_dim,
    #     ch_mult=(1, 1, 2, 2, 4),
    #     ckpt_path=args.rgb_vae_path
    # )
    # vae2d.eval()  # Set to evaluation mode, do not update parameters
    vae2d = AutoencoderKL.from_pretrained("ostris/vae-kl-f8-d16").eval()
    
    model_func = globals()[args.model]
    model = model_func(
        img_h=args.img_h,
        img_w=args.img_w,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.rgb_vae_embed_dim,
        mask_ratio_min=args.mask_ratio_min,
        condition_drop_prob=args.condition_drop_prob,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        buffer_size=args.buffer_size,
        diffloss_d=args.diffloss_d,
        diffloss_w=args.diffloss_w,
        num_sampling_steps=args.num_sampling_steps,
        diffusion_batch_mul=args.diffusion_batch_mul,
        grad_checkpointing=args.grad_checkpointing,
    )

    
    # Load diffusion model checkpoint (if provided)
    if args.diffusion_ckpt and os.path.isfile(args.diffusion_ckpt):
        print(f"Loading diffusion model checkpoint from {args.diffusion_ckpt}")
        checkpoint = torch.load(args.diffusion_ckpt, map_location='cpu')
        
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
        
        # Create new state dict, excluding keys to be ignored
        ckpt_state_dict = {k: v for k, v in ckpt_state_dict.items() 
                               if not any(pos_key in k for pos_key in pos_embed_keys)}
        
        # Use strict=False to automatically handle unmatched parameters
        missing_keys, unexpected_keys = model.load_state_dict(ckpt_state_dict, strict=False)
        
        print(f"Loading complete!")
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        
    for param in vae2d.parameters():
        param.requires_grad = False
    # print the trainable parameters and relative ratio
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}, Total parameters: {total_params}, Trainable ratio: {trainable_params/total_params:.2%}")
    
    
    return model, vae2d

def load_data(args):
    """Load dataset"""
    print("Loading dataset...")
    data_loader = get_online_process_dataloader(
        image_folder=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        shuffle=True,
        distributed=args.distributed,
        rank=args.rank,
        world_size=args.world_size,
        # Image processing parameters
        kernel_size=args.kernel_size,
        blur_intensity=args.blur_intensity,
        blur_samples=args.blur_samples
    )
    
    dataset_size = len(data_loader.dataset)
    if dataset_size == 0:
        raise ValueError(f"Dataset is empty, please check data path: {args.data_root}")
    
    print(f"Dataset loaded with {dataset_size} samples")
    
    # Create spike stream converter
    spike_converter = SpikeConverter(
        photon_samples=args.photon_samples,
        target_coverage=args.target_coverage,
        smooth_sigma=args.smooth_sigma,
        gamma=args.gamma
    )
    
    # Get one sample to set frame resolution
    data = next(iter(data_loader))
    # Generate spike data using spike converter
    spike_data = spike_converter.rgb_to_spike(data['rgb'])
    args.frame_resolution = spike_data.shape[1]
    
    # Print data shapes for verification
    print(f"Data shape verification:")
    print(f"- RGB image: {data['rgb'].shape}")
    print(f"- Blurred image: {data['blur'].shape}")
    print(f"- Spike stream data: {spike_data.shape}")
    
    return data_loader, spike_converter

def unwrap_model(model, distributed=True):
    """Get original model from DistributedDataParallel wrapper"""
    if distributed and hasattr(model, 'module'):
        return model.module
    return model



def compute_metric(rgb_gt, generated_images):
    # calculate the metric
    metrics = {}
    method_list = ['mar']
    metric_list = ['mse','ssim','psnr','lpips']
    for method_name in method_list:
        metrics[method_name] = {}  # initialize each method's dictionary
        for metric_name in metric_list:
            metrics[method_name][metric_name] = AverageMeter()
    for key in metric_list :
        # metrics['TFP'][key].update(compute_img_metric(tfp,sharp,key))
        metrics['mar'][key].update(compute_img_metric(rgb_gt, generated_images, key))
    # Print all results
    output_path = args.output_dir + "/GOPRO_metric.txt"
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
    # if not misc.is_main_process():
    #     return
    print("evaluate ")
    model.eval()
    
    # get the original model (handle distributed training)
    if args.distributed:
        model_unwrapped = model.module
    else:
        model_unwrapped = model
    
    # get samples from dataset for visualization
    blur_images = viz_data['blur'].to(device)[:8]  # only use first 4 samples
    rgb_gt = viz_data['rgb'].to(device)[:8]
    
    # directly generate spike data on GPU
    spike_stream = spike_converter.rgb_to_spike(rgb_gt)[:8]
    
    
    # directory to save output images
    output_dir = Path(args.output_dir) / 'generations'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # get latent representation via VAE encoder
    with torch.no_grad():
        # encode blurry RGB
        # rgb_blur_posterior = vae2d.encode(blur_images)
        # rgb_blur_latent = rgb_blur_posterior.sample().mul_(args.vae_scale)
        rgb_blur_latent = vae2d.encode(blur_images).latent_dist.sample().mul_(args.vae_scale)
        vae_recon_blur = vae2d.decode(rgb_blur_latent / args.vae_scale)[0]
        rgb_gt_latent = vae2d.encode(rgb_gt).latent_dist.sample().mul_(args.vae_scale)
        vae_recon_gt = vae2d.decode(rgb_gt_latent / args.vae_scale)[0]
        # # encode spike stream
        # condition = condition_unwrapped(spike_data_3d, blur_images)
        
        # # convert latent to token
        # rgb_blur_tokens = model_unwrapped.patchify(rgb_blur_latent)
        # condition_tokens = model_unwrapped.patchify(condition)
        
        # generate samples
        generated_tokens = model_unwrapped.forward_image(
            rgb_blur_latent, 
            spike_stream=spike_stream,
            temperature=0.5
        )
        
        no_spike_generated_tokens = model_unwrapped.forward_image(
            rgb_blur_latent, 
            spike_stream=None,
            cfg = 4.0, 
            cfg_schedule="constant",
            temperature=0.5
        )
        
        no_rgb_generated_tokens = model_unwrapped.forward_image(
            blur_latent=None, 
            spike_stream=spike_stream,
            cfg = 4.0, 
            cfg_schedule="constant",
            temperature=0.5
        )
        
        # decode the generated latent representation
        # generated_images = vae2d.decode(generated_tokens / args.vae_scale)
        generated_images = vae2d.decode(generated_tokens / args.vae_scale)[0]
        no_spike_generated_images = vae2d.decode(no_spike_generated_tokens / args.vae_scale)[0]
        no_rgb_generated_images = vae2d.decode(no_rgb_generated_tokens / args.vae_scale)[0]
        # normalize to valid range
        blur_images = (blur_images + 1) / 2  # convert from [-1,1] to [0,1]
        rgb_gt = (rgb_gt + 1) / 2  # convert from [-1,1] to [0,1]
        generated_images = (generated_images + 1) / 2  # convert from [-1,1] to [0,1]
        no_spike_generated_images = (no_spike_generated_images + 1) / 2  # convert from [-1,1] to [0,1]
        no_rgb_generated_images = (no_rgb_generated_images + 1) / 2  # convert from [-1,1] to [0,1]
        vae_recon_blur = (vae_recon_blur + 1) / 2  # convert from [-1,1] to [0,1]
        vae_recon_gt = (vae_recon_gt + 1) / 2  # convert from [-1,1] to [0,1]
        # pick a spike frame from spike stream
        compute_metric(rgb_gt, generated_images)
        spike_frame = rgb_gt*spike_stream[:, :3, :, :]

    # create summary comparison image
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
    
    # save comparison image
    plt.figure(figsize=(32, 32))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(str(output_dir / f'comparison_epoch_{epoch}_{step}.png'))
    plt.close()
    
    # restore training mode
    model.train()


def parse_and_average_metrics(file_path):
    mse_list = []
    ssim_list = []
    psnr_list = []
    lpips_list = []

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            mse = float(parts[2])
            ssim = float(parts[4])
            psnr = float(parts[6])
            lpips = float(parts[8])
            
            mse_list.append(mse)
            ssim_list.append(ssim)
            psnr_list.append(psnr)
            lpips_list.append(lpips)

    avg_mse = sum(mse_list) / len(mse_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)
    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_lpips = sum(lpips_list) / len(lpips_list)

    return {
        'avg_mse': avg_mse,
        'avg_ssim': avg_ssim,
        'avg_psnr': avg_psnr,
        'avg_lpips': avg_lpips
    }


def main(args):
    # initialize distributed training
    misc.init_distributed_mode(args)
    
    print('Working directory: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    
    device = torch.device(args.device)
    
    # fix random seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    
    # setup data loader
    data_loader, spike_converter = load_data(args)
    
    # load model
    model, vae2d = load_model(args)
    
    # move to device
    model.to(device)
    vae2d.to(device)
    
    # distributed training setup
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu],
            find_unused_parameters=True
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.online_eval:
            torch.cuda.empty_cache()  # clear cache to prevent OOM
            output_path = args.output_dir + "/metrics.txt"
            if dist.get_rank() == 0:
                os.system(f"rm {output_path}")
            dist.barrier()
            # import pdb; pdb.set_trace()
            with torch.no_grad():
                for batch_idx, viz_data in tqdm(enumerate(data_loader), total=len(data_loader), desc="Eval Test Dataset"):
                    evaluate(model, vae2d, viz_data, device, epoch, batch_idx, log_writer, args, spike_converter)
                    torch.cuda.empty_cache()
                    # break

            # wait for all GPUs to reach this point
            dist.barrier()
            if dist.get_rank() == 0:

                averages = parse_and_average_metrics(output_path)
                with open(args.output_dir + "/mean_metric.txt", 'a') as f:
                    for key, value in averages.items():
                        f.write(f'{key}: {value:.4f}\n')
            # viz_data = next(iter(data_loader))
            # evaluate(model, vae2d, viz_data, device, epoch, -1, log_writer, args, spike_converter)
            torch.cuda.empty_cache()
        
        # ensure log flushing (only in main process)
        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()
    
    # compute total training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if misc.is_main_process():
        print(f'Total training time: {total_time_str}')

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    
    # create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    
    print(f"Batch size: {args.batch_size}")
    
    main(args)
