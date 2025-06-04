from functools import partial

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import Block

from models.diffloss import DiffLoss

def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking


# Add RoPE positional encoding implementation
def get_rotary_embedding(seq_len, dim, base=10000, device=None):
    """Generate rotary positional embedding"""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    position = torch.arange(seq_len, device=device).float()
    sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
    return torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)


def apply_rotary_pos_emb(x, pos_emb):
    """Apply rotary positional embedding to input tensor"""
    x_rope, x_pass = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    x_rope = (x_rope * pos_emb[..., :x_rope.shape[-1]]) + (x_pass * pos_emb[..., x_pass.shape[-1]:])
    x_pass = (x_pass * pos_emb[..., :x_pass.shape[-1]]) - (x_rope * pos_emb[..., x_rope.shape[-1]:])
    return torch.cat((x_rope, x_pass), dim=-1)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding module with support for extrapolation"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.device = device
        
        # Precompute positional encodings
        self.register_buffer("inv_freq", None, persistent=False)
        self._update_cos_sin_cache()
    
    def _update_cos_sin_cache(self):
        """Update sine and cosine cache"""
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(self.device) / self.dim))
        position = torch.arange(self.max_position_embeddings, device=self.device).float()
        sinusoid_inp = torch.einsum("i,j->ij", position, self.inv_freq)
        self.register_buffer("cos_cached", sinusoid_inp.cos(), persistent=False)
        self.register_buffer("sin_cached", sinusoid_inp.sin(), persistent=False)
    
    def forward(self, x, seq_len=None):
        """
        Apply rotary positional embedding
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            seq_len: Sequence length, if None uses length from x
            
        Returns:
            Tensor with positional embedding applied
        """
        if seq_len is None:
            seq_len = x.shape[1]
        
        # Dynamically compute if sequence length exceeds cache
        if seq_len > self.max_position_embeddings:
            position = torch.arange(seq_len, device=x.device).float()
            sinusoid_inp = torch.einsum("i,j->ij", position, self.inv_freq)
            cos = sinusoid_inp.cos()
            sin = sinusoid_inp.sin()
        else:
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        
        # Broadcast sine and cosine to batch dimension
        cos = cos.unsqueeze(0)  # [1, seq_len, dim/2]
        sin = sin.unsqueeze(0)  # [1, seq_len, dim/2]
        
        # Split input tensor in half
        x_rope, x_pass = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        
        # Apply rotary positional embedding
        x_rope = (x_rope * cos) + (x_pass * sin)
        x_pass = (x_pass * cos) - (x_rope * sin)
        
        # Merge result
        return torch.cat((x_rope, x_pass), dim=-1)


class SpikeStreamProcessor(nn.Module):
    """
    Lightweight Spike Stream processing module
    
    Converts spike stream data in shape [B, 1, T, H, W] to feature representation of shape [B, 512, T/8, H/8, W/8]
    Uses a 3D convolutional network to fuse spatiotemporal information and perform downsampling
    """
    def __init__(self, out_channels=512, use_residual=True):
        super().__init__()
        self.out_channels = out_channels
        self.use_residual = use_residual
        
        # Initial feature extraction
        self.init_conv = nn.Conv3d(
            in_channels=1, 
            out_channels=32, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.init_act = nn.LeakyReLU(0.2, inplace=True)
        self.init_norm = nn.InstanceNorm3d(32)
        
        # First downsampling block: T/2, H/2, W/2, 32->64 channels
        self.down1_conv = nn.Conv3d(
            in_channels=32, 
            out_channels=64, 
            kernel_size=3, 
            stride=2, 
            padding=1
        )
        self.down1_act = nn.LeakyReLU(0.2, inplace=True)
        self.down1_norm = nn.InstanceNorm3d(64)
        
        # Second downsampling block: T/4, H/4, W/4, 64->128 channels
        self.down2_conv = nn.Conv3d(
            in_channels=64, 
            out_channels=128, 
            kernel_size=3, 
            stride=2, 
            padding=1
        )
        self.down2_act = nn.LeakyReLU(0.2, inplace=True)
        self.down2_norm = nn.InstanceNorm3d(128)
        
        # Third downsampling block: T/8, H/8, W/8, 128->256 channels
        self.down3_conv = nn.Conv3d(
            in_channels=128, 
            out_channels=256, 
            kernel_size=3, 
            stride=2, 
            padding=1
        )
        self.down3_act = nn.LeakyReLU(0.2, inplace=True)
        self.down3_norm = nn.InstanceNorm3d(256)
        
        # Final channel adjustment to target dimension
        self.final_conv = nn.Conv3d(
            in_channels=256, 
            out_channels=out_channels, 
            kernel_size=1, 
            stride=1
        )
        self.final_act = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Spike Stream data of shape [B, 1, T, H, W]
            
        Returns:
            Feature representation with fused spatiotemporal information, shape [B, 512, T/8, H/8, W/8]
        """
        # Initial feature extraction
        x = self.init_conv(x)
        x = self.init_act(x)
        x = self.init_norm(x)
        init_features = x
        
        # First downsampling block
        x = self.down1_conv(x)
        x = self.down1_act(x)
        x = self.down1_norm(x)
        down1_features = x
        
        # Second downsampling block
        x = self.down2_conv(x)
        x = self.down2_act(x)
        x = self.down2_norm(x)
        down2_features = x
        
        # Third downsampling block
        x = self.down3_conv(x)
        x = self.down3_act(x)
        x = self.down3_norm(x)
        
        # Final channel adjustment
        x = self.final_conv(x)
        x = self.final_act(x)
        
        return x



class SpikeTemporalFuser(nn.Module):
    """
    Spike stream temporal fusion module
    
    Compresses features of shape [B, 512, T/8, H/8, W/8] into [B, 512, H/8, W/8],
    using attention mechanism along the temporal dimension
    """
    def __init__(self, channels=512, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        # Generate attention weights along temporal dimension
        self.time_attention = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final adjustment
        self.fusion_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.fusion_norm = nn.LayerNorm([channels])
        self.fusion_act = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Feature representation, shape [B, C, T, H, W]
            
        Returns:
            Features after temporal fusion, shape [B, C, H, W]
        """
        B, C, T, H, W = x.shape
        
        # Generate attention weights over time
        time_weights = self.time_attention(x)  # [B, C, T, H, W]
        
        # Apply temporal attention weights
        weighted_features = x * time_weights  # [B, C, T, H, W]
        
        # Fuse along temporal dimension
        fused_features = torch.sum(weighted_features, dim=2)  # [B, C, H, W]
        
        # Final adjustment
        fused_features = self.fusion_conv(fused_features)
        fused_features = fused_features.permute(0, 2, 3, 1)  # [B, H, W, C]
        fused_features = self.fusion_norm(fused_features)
        fused_features = fused_features.permute(0, 3, 1, 2)  # [B, C, H, W]
        fused_features = self.fusion_act(fused_features)
        
        return fused_features


class SpikeFeatExtractor(nn.Module):
    """
    Spike feature extractor, combining SpikeStreamProcessor and SpikeTemporalFuser
    
    Converts spike stream data of shape [B, 1, T, H, W] to feature representation [B, 512, H/8, W/8]
    """
    def __init__(self, out_channels=512, include_temporal_fusion=True):
        super().__init__()
        self.out_channels = out_channels
        self.include_temporal_fusion = include_temporal_fusion
        
        # Spike stream processing
        self.spike_processor = SpikeStreamProcessor(out_channels=out_channels)
        
        # Temporal fusion (optional)
        if include_temporal_fusion:
            self.temporal_fuser = SpikeTemporalFuser(channels=out_channels)
    
    def forward(self, spike_stream):
        """
        Forward pass
        
        Args:
            spike_stream: Spike stream data, shape [B, 1, T, H, W] or [B, T, H, W]
            
        Returns:
            Fused feature representation, shape [B, C, H/8, W/8] if include_temporal_fusion=True,
            otherwise [B, C, T/8, H/8, W/8]
        """
        # Ensure input has channel dimension
        if spike_stream.dim() == 4:
            spike_stream = spike_stream.unsqueeze(1)  # [B, T, H, W] -> [B, 1, T, H, W]
            
        # Process spike stream
        features = self.spike_processor(spike_stream)  # [B, C, T/8, H/8, W/8]
        
        # Temporal fusion (optional)
        if self.include_temporal_fusion:
            features = self.temporal_fuser(features)  # [B, C, H/8, W/8]
            
        return features


class SpikeMAR(nn.Module):
    """ MAR model combining Spike Stream and blurred RGB, retaining original class embedding injection """
    def __init__(self, img_h=256, img_w=256, vae_stride=8, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=16,
                 mask_ratio_min=0.7,
                 label_drop_prob=0.1,
                 class_num=1000,
                 condition_drop_prob=0.1,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patch related
        self.vae_embed_dim = vae_embed_dim
        self.token_embed_dim = vae_embed_dim * patch_size**2
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.img_h = img_h
        self.img_w = img_w
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = img_h // vae_stride // patch_size
        self.seq_w = img_w // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w 
        self.grad_checkpointing = grad_checkpointing
        
        # --------------------------------------------------------------------------
        # Class Embedding (from MAR)
        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # --------------------------------------------------------------------------
        # Spike Stream processor
        self.spike_processor = SpikeFeatExtractor(
            out_channels=encoder_embed_dim,
            include_temporal_fusion=True
        )
        
        # Spike feature projection layer
        self.spike_proj = nn.Linear(encoder_embed_dim, encoder_embed_dim, bias=True)
        # zero init
        # self.spike_proj.weight.data.zero_()
        # self.spike_proj.bias.data.zero_()
        
        # --------------------------------------------------------------------------
        # MAR variant mask ratio, Gaussian truncated around 1.0
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)
        # --------------------------------------------------------------------------
        # MAR encoder
        self.encoder_depth = encoder_depth
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        
        # Replace original positional embedding with RoPE
        self.encoder_rope = RotaryPositionalEmbedding(
            dim=encoder_embed_dim,
            max_position_embeddings=2048,
            base=10000
        )
        
        # Keep original learned positional embedding as backup
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, 320, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # MAR decoder
        self.decoder_depth = decoder_depth
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Replace original positional embedding with RoPE
        self.decoder_rope = RotaryPositionalEmbedding(
            dim=decoder_embed_dim,
            max_position_embeddings=2048,
            base=10000
        )
        
        # Keep original learned positional embedding as backup
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, 320, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        
        # Replace original positional embedding with RoPE
        self.diffusion_rope = RotaryPositionalEmbedding(
            dim=decoder_embed_dim,
            max_position_embeddings=2048,
            base=10000
        )
        
        # Keep original learned positional embedding as backup
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, 256, decoder_embed_dim))

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing
        )
        self.diffusion_batch_mul = diffusion_batch_mul

    def initialize_weights(self):
        # Initialize parameters
        # torch.nn.init.normal_(self.fake_condition, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        # Initialize linear layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        """Split image into patch tokens"""
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def sample_orders(self, bsz):
        # Generate random orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        # Generate random masks
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def forward_mae_encoder(self, x, mask):
        """MAE encoder forward pass with per-block conditional injection"""
        bsz, seq_len, embed_dim = x.shape
        # ensure mask as float
        mask = mask.float()
        
        # Use RoPE positional embedding instead of learned
        x = self.encoder_rope(x)
        
        # Backup: x = x + self.encoder_pos_embed_learned
        
        x = self.z_proj_ln(x)
        # Keep unmasked tokens
        x = x[(1-mask).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)
        
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x)
        else:
            for i in range(self.encoder_depth):
                x = self.encoder_blocks[i](x)
        x = self.encoder_norm(x)
        
        return x

    def forward_mae_decoder(self, x, mask):
        # Project to decoder dimension
        x = self.decoder_embed(x)
        mask = mask.float()
        # Add mask tokens
        mask_tokens = self.mask_token.repeat(mask.shape[0], mask.shape[1], 1).to(x.dtype)
        x_visible = mask_tokens.clone()
        x_visible[(1 - mask).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        
        # Use RoPE positional embedding instead of learned
        x = self.decoder_rope(x_visible)

        # Apply decoder transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x)
        else:
            for i in range(self.decoder_depth):
                x = self.decoder_blocks[i](x)
        x = self.decoder_norm(x)
        
        # Use RoPE positional embedding instead of learned
        x = self.diffusion_rope(x)
        
        # Backup: x = x + self.diffusion_pos_embed_learned
        
        return x

    def forward_loss(self, z, target, mask=None):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul) if mask is not None else None
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss


    def forward(self, gt_latent, blur_latent=None, spike_stream=None, reture_z=True):
        """
        Forward pass function
        
        Args:
            blur_latent (torch.Tensor): Latent representation of the blurred image, shape [B, C, H_vae, W_vae]
            gt_latent (torch.Tensor): Latent representation of the ground truth image, shape [B, C, H_vae, W_vae]
            image_mask (torch.Tensor, optional): Mask in original image space, shape [B, H_img, W_img]
            spike_stream (torch.Tensor, optional): Spike stream data, shape [B, 1, T, H, W]
        
        Returns:
            torch.Tensor: diffusion loss
        """
        if blur_latent is None and spike_stream is None:
            raise ValueError("blur_latent and spike_stream cannot both be None")
        
        # Convert VAE latent to token representation
        gt_tokens = self.patchify(gt_latent)
        bsz, seq_len, _ = gt_tokens.shape
        
        if blur_latent is not None:
            blur_tokens = self.patchify(blur_latent)
            blur_tokens = self.z_proj(blur_tokens)
            mixture_ratio = 0
        else:
            blur_tokens = 0
            
        if spike_stream is not None:
            # Process spike stream data to get feature representation
            spike_latents = self.spike_processor(spike_stream)  # [B, C, H/8, W/8]
            spike_tokens = self.patchify(spike_latents)
            spike_tokens = self.spike_proj(spike_tokens)
            mixture_ratio = 1
        else:
            spike_tokens = 0
        
        # Assign a random mixture ratio of spike and blur
        if blur_latent is not None and spike_stream is not None:
            mixture_ratio = torch.rand(bsz, 1).to(blur_tokens.device).unsqueeze(-1)
        
        mixture_ratio = 0.5
        
        tokens = blur_tokens * (1 - mixture_ratio) + spike_tokens * mixture_ratio
        zero_mask = torch.zeros(bsz, seq_len).cuda()
        one_mask = torch.ones(bsz, seq_len).cuda()

        # MAE encoder
        x = self.forward_mae_encoder(x=tokens, mask=zero_mask)
            
        # MAE decoder
        z = self.forward_mae_decoder(x, zero_mask)
        
        # Compute loss
        diffloss = self.forward_loss(z=z, target=gt_tokens, mask=None)
        
        if reture_z:
            return diffloss, z
        else:
            return diffloss

    def recon_tokens(self,  blur_latent=None, spike_stream=None, gamma=0.5, temperature=1.0):
        # import pdb; pdb.set_trace()
        if blur_latent is None and spike_stream is None:
            raise ValueError("blur_latent and spike_stream cannot both be None")
        if blur_latent is not None:
            blur_tokens = self.patchify(blur_latent).cuda()
            blur_tokens = self.z_proj(blur_tokens)
        else:
            blur_tokens = 0
        if spike_stream is not None:
            # Process spike stream data to get feature representation
            spike_latents = self.spike_processor(spike_stream)  # [B, C, H/8, W/8]
            spike_tokens = self.patchify(spike_latents)
            spike_tokens = self.spike_proj(spike_tokens)
        else:
            spike_tokens = 0
        # import pdb; pdb.set_trace()
        tokens = blur_tokens * (1 - gamma) + spike_tokens * gamma
        bsz, seq_len, _ = tokens.shape
        mask = torch.zeros(bsz, seq_len).cuda()
        x = self.forward_mae_encoder(tokens, mask)
        z = self.forward_mae_decoder(x, mask)
        z = z.reshape(-1, z.shape[-1])
        sampled_token = self.diffloss.sample(z, temperature=temperature)
        if sampled_token.shape[0] != bsz:
            sampled_token = sampled_token.reshape(bsz, -1, self.token_embed_dim)
        latents = self.unpatchify(sampled_token)
        return latents

    def sample_tokens(self, condition_tokens, rgb_blur_tokens, num_iter=64, temperature=1.0, progress=False):
        """
        Sample tokens for generation
        
        Args:
            condition_tokens (torch.Tensor): Token representation of condition
            rgb_blur_tokens (torch.Tensor): Token representation of blurred RGB
            num_iter (int): Number of iterations
            temperature (float): Temperature parameter
            progress (bool): Whether to show progress bar
        
        Returns:
            torch.Tensor: Generated tokens
        """
        
        # Initialize mask and tokens
        bsz = rgb_blur_tokens.shape[0]
        mask = torch.ones(bsz, self.seq_len).cuda()
        # tokens = rgb_blur_tokens.clone()
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
        orders = self.sample_orders(bsz)
        
        indices = list(range(num_iter))
        
        if progress:
            indices = tqdm(indices)
            
        # Generate latents
        for step in indices:
            cur_tokens = tokens.clone()
            # MAE encoder
            x = self.forward_mae_encoder(tokens, mask, condition_tokens)
            # MAE decoder
            z = self.forward_mae_decoder(x, mask, condition_tokens)
            # Mask ratio for the next round, following MaskGIT and MAGE
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()

            # Mask out at least one for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

            # Get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            
            # Sample token latents
            z_pred = z[mask_to_pred.nonzero(as_tuple=True)]
            sampled_token_latent = self.diffloss.sample(z_pred, temperature)
            
            # Update tokens
            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()
        # unpatchify
        tokens = self.unpatchify(tokens)
        return tokens


def spike_mar_base(**kwargs):
    model = SpikeMAR(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def spike_mar_large(**kwargs):
    model = SpikeMAR(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def spike_mar_huge(**kwargs):
    model = SpikeMAR(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
