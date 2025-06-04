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





def get_args_parser():
    parser = argparse.ArgumentParser('SpikeGen training', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int,
                      help='batch size per GPU')
    parser.add_argument('--epochs', default=100, type=int)
    
    # Model parameters
    parser.add_argument('--model', default='spike_mar_large', type=str, metavar='MODEL',
                      help='model name')

    # VAE parameters
    parser.add_argument('--img_h', default=360, type=int,
                      help='input image height')
    parser.add_argument('--img_w', default=640, type=int,
                      help='input image width')
    parser.add_argument('--rgb_vae_path', default="ostris/vae-kl-f8-d16", type=str,
                      help='pretrained RGB VAE path')
    parser.add_argument('--rgb_vae_embed_dim', default=16, type=int,
                      help='RGB VAE embed dim')
    parser.add_argument('--vae_stride', default=8, type=int,
                      help='tokenizer stride')
    parser.add_argument('--patch_size', default=1, type=int,
                      help='number of tokens in a patch')
    parser.add_argument('--vae_scale', default=0.2325, type=float,
                      help='VAE scale factor')
    
    # Generation parameters
    parser.add_argument('--eval_freq', type=int, default=1,
                      help='evaluation frequency')
    parser.add_argument('--save_last_freq', type=int, default=5,
                      help='save last frequency')
    parser.add_argument('--online_eval', action='store_true')

    
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02,
                      help='weight decay')
    parser.add_argument('--grad_checkpointing', action='store_true')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                      help='learning rate')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                      help='base learning rate')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                      help='minimum learning rate')
    parser.add_argument('--lr_schedule', type=str, default='cosine',
                      help='learning rate scheduler type')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                      help='number of warmup epochs')
    
    # MAR parameters
    parser.add_argument('--diffusion_ckpt', default='pretrained_models/mar/mar_base/checkpoint-last.pth', type=str,
                      help='Diffusion model checkpoint path')
    parser.add_argument('--grad_clip', type=float, default=3.0,
                      help='gradient clipping')
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                      help='attention dropout')
    parser.add_argument('--proj_dropout', type=float, default=0.1,
                      help='projection dropout')
    
    # Diffusion Loss parameters
    parser.add_argument('--diffloss_d', type=int, default=6)
    parser.add_argument('--diffloss_w', type=int, default=1024)
    parser.add_argument('--num_sampling_steps', type=str, default="100")
    parser.add_argument('--diffusion_batch_mul', type=int, default=1)
    parser.add_argument('--temperature', default=1.0, type=float,
                      help='diffusion loss sampling temperature')
    
    # Dataset parameters
    parser.add_argument('--data_root', default='./datasets/images',
                      help='image folder path')
    parser.add_argument('--scenes', type=str, nargs='+', default=None,
                      help='scene name list')
    
    parser.add_argument('--output_dir', default='./output_dir',
                      help='save path')
    parser.add_argument('--log_dir', default='./output_dir',
                      help='tensorboard log path')
    parser.add_argument('--device', default='cuda',
                      help='device')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                      help='resume from checkpoint')
    
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                      help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                      help='pin memory')
    parser.set_defaults(pin_mem=True)
    
    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                      help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                      help='distributed training url')
    
    # Image processing parameters
    parser.add_argument('--kernel_size', type=int, default=40,
                      help='blur kernel size')
    parser.add_argument('--blur_intensity', type=float, default=40,
                      help='blur intensity')
    parser.add_argument('--blur_samples', type=int, default=8,
                      help='blur samples')
    parser.add_argument('--photon_samples', type=int, default=8,
                      help='photon samples')
    parser.add_argument('--target_coverage', type=float, default=0.1,
                      help='target sampling coverage')
    parser.add_argument('--smooth_sigma', type=float, default=1,
                      help='smooth sigma')
    parser.add_argument('--gamma', type=float, default=1,
                      help='gamma correction')
    
    return parser

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

def train_one_epoch(model, vae2d, data_loader, optimizer, 
                   device, epoch, loss_scaler, log_writer=None, args=None, spike_converter=None):
    
    def forward_pixel_loss(args, vae, model, spike_converter, gt_rgb, gt_spike_stream, z):
        if args.distributed:
            model_unwrapped = model.module
        else:
            model_unwrapped = model
    
        bsz, seq_len, token_embed_dim = z.shape
        one_mask = torch.ones_like(z)
        z = z.reshape(-1, token_embed_dim)
        
        # Sampling token using the model
        sampled_token = model_unwrapped.diffloss.sample(z, temperature=0.5)
        
        if sampled_token.shape[0] != bsz:
            sampled_token = sampled_token.reshape(bsz, seq_len, -1)
        
        latents = model_unwrapped.unpatchify(sampled_token)
        pred_rgb = vae.decode(latents / args.vae_scale)[0]
        
        # Convert RGB to spike train using the spike_converter
        pred_spike_stream = spike_converter.rgb_to_spike(pred_rgb)

        # Calculate the Spikemax loss
        spikeloss = torch.nn.functional.mse_loss(pred_spike_stream.sum(dim=1),
                                                 gt_spike_stream.sum(dim=1))  
        
        # Calculate the Mean Squared Error (MSE) loss for RGB images
        mseloss = torch.nn.functional.mse_loss(pred_rgb, gt_rgb)
        
        return spikeloss, mseloss
    
    """Train one epoch"""
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'
    print_freq = 20
    
    # Ensure consistent order in distributed training
    if args.distributed and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)
    
    # Initialize metrics
    metric_logger.update(loss=0.0)
    current_lr = optimizer.param_groups[0]["lr"]
    metric_logger.update(lr=current_lr)
        
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if  data_iter_step % args.eval_freq == 0:
            torch.cuda.empty_cache()
            evaluate(model, vae2d, batch, device, epoch, data_iter_step, log_writer, args, spike_converter)
            torch.cuda.empty_cache()
        # Update learning rate
        if args.lr_schedule == 'cosine':
            progress = data_iter_step / len(data_loader) + epoch
            if epoch < args.warmup_epochs:
                current_lr = args.lr * progress / args.warmup_epochs
            else:
                progress = (progress - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                current_lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
        blur_images = batch['blur'].to(device, non_blocking=True)
        gt_images = batch['rgb'].to(device, non_blocking=True)
        gt_spike_stream = spike_converter.rgb_to_spike(gt_images)
        with torch.no_grad():
            rgb_blur_latent = vae2d.encode(blur_images).latent_dist.sample().mul_(args.vae_scale)
            rgb_gt_latent = vae2d.encode(gt_images).latent_dist.sample().mul_(args.vae_scale)
        with torch.cuda.amp.autocast():
            diffloss, z = model(rgb_gt_latent,
                                rgb_blur_latent,
                                spike_stream=gt_spike_stream,
                                reture_z=True,
                                gamma=0.5
                                )
            spikeloss, mseloss = forward_pixel_loss(args, vae2d, model, spike_converter, gt_images, gt_spike_stream, z)
        # Log loss values (for logs)
        diffloss_value = diffloss.item()
        spikeloss_value = spikeloss.item()
        mseloss_value = mseloss.item()
        loss_value = diffloss_value + spikeloss_value + mseloss_value
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        loss = diffloss + spikeloss + mseloss
        loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(diffloss=diffloss_value)
        metric_logger.update(spikeloss=spikeloss_value)
        metric_logger.update(mseloss=mseloss_value)
        metric_logger.update(lr=current_lr)

        if log_writer is not None and misc.is_main_process():
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value, epoch_1000x)
            log_writer.add_scalar('diffloss', diffloss_value, epoch_1000x)
            log_writer.add_scalar('spikeloss', spikeloss_value, epoch_1000x)
            log_writer.add_scalar('mseloss', mseloss_value, epoch_1000x)
            log_writer.add_scalar('lr', current_lr, epoch_1000x)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

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
    
    # Get the raw model (handle distributed training case)
    if args.distributed:
        model_unwrapped = model.module
    else:
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

def main(args):
    # Initialize distributed training
    misc.init_distributed_mode(args)
    
    print('Working directory: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    
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
    
    # Distributed training setup
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu],
            find_unused_parameters=True
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    
    # Compute model parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if misc.is_main_process():
        print(f"Number of parameters: {n_params/1e6:.2f}M")
    
    eff_batch_size = args.batch_size * misc.get_world_size()
    
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    
    if misc.is_main_process():
        print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
        print("actual lr: %.2e" % args.lr)
        print("effective batch size: %d" % eff_batch_size)
    
    # Set weight decay - no decay for bias and norm layers
    param_groups = []
    # Apply weight decay directly to MAR model
    mar_param_groups = misc.add_weight_decay(
        model_without_ddp, 
        weight_decay=args.weight_decay
    )
    param_groups.extend(mar_param_groups)
    
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    if misc.is_main_process():
        print(f"Optimizer: {optimizer}")
    
    # Gradient scaler
    loss_scaler = NativeScaler()
    
    # Resume training from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Resuming model from {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu')
            
            model_without_ddp.load_state_dict(checkpoint['model'])
            
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch'] + 1
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print(f"Resume complete! Training from epoch {args.start_epoch}.")
    
    # Create TensorBoard writer (only in main process)
    if misc.is_main_process() and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    
    # Start training
    if misc.is_main_process():
        print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        # Train one epoch
        train_stats = train_one_epoch(
            model, vae2d,
            train_loader,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args,
            spike_converter=spike_converter
        )
        
        # Save checkpoint (only in main process)
        if misc.is_main_process() and (epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs):
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{epoch}.pth") 
            torch.save({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
                'scaler': loss_scaler.state_dict(),
            }, checkpoint_path)
                
            # Also save as checkpoint-last.pth
            last_path = os.path.join(args.output_dir, "checkpoint-last.pth")
            torch.save({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
                'scaler': loss_scaler.state_dict(),
            }, last_path)
        
        # Online evaluation (only in main process)
        if args.online_eval:
            torch.cuda.empty_cache()  # Clear cache to avoid OOM
            viz_data = next(iter(train_loader))
            evaluate(model, vae2d, viz_data, device, epoch, -1, log_writer, args, spike_converter)
            torch.cuda.empty_cache()
        
        # Ensure logs are flushed (only in main process)
        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()
    
    # Calculate total training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if misc.is_main_process():
        print(f'Total training time: {total_time_str}')

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir
    
    # Compute actual learning rate
    if args.lr is None:
        args.lr = args.blr * args.batch_size / 256
    
    print(f"Base learning rate: {args.blr}")
    print(f"Actual learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    
    main(args)