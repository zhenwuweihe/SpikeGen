import os
import torch
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
import glob
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F




class SpikeConverter:
    """Spike Stream Converter

    Converts RGB images to spike stream data
    """
    
    def __init__(self, 
                 photon_samples=8,
                 target_coverage=0.1,
                 smooth_sigma=1,
                 gamma=1):
        """Initialize Spike Stream Converter"""
        self.photon_samples = photon_samples
        self.target_coverage = target_coverage
        self.smooth_sigma = smooth_sigma
        self.gamma = gamma
    
    def get_photon_probability_map(self, rgb_image):
        """Generate probability map for photon sampling
        
        Args:
            rgb_image: RGB image with shape [C, H, W] or [B, C, H, W], values in [0, 1]
            
        Returns:
            prob_map: probability map, shape [H, W] or [B, H, W], values in [0, 1]
        """
        # Ensure input is a PyTorch tensor
        if not isinstance(rgb_image, torch.Tensor):
            rgb_image = torch.from_numpy(rgb_image).float()
        
        if rgb_image.dim() == 4:  # [B, C, H, W]
            batch_size = rgb_image.shape[0]
            
            intensity = rgb_image[:, 0, :, :] * 0.2989 + rgb_image[:, 1, :, :] * 0.5870 + rgb_image[:, 2, :, :] * 0.1140  # [B, H, W]
            # norm to 0-1 by min-max
            intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
            # Create probability map - using Gaussian blur
            if self.smooth_sigma > 0:
                # Use PyTorch's Gaussian blur
                kernel_size = int(2 * round(3 * self.smooth_sigma) + 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                coords = torch.arange(kernel_size, dtype=torch.float32, device=rgb_image.device) - (kernel_size - 1) / 2
                x, y = torch.meshgrid(coords, coords, indexing='ij')
                gaussian_kernel = torch.exp(-(x**2 + y**2) / (2 * self.smooth_sigma**2))
                gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
                gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
                
                # Apply Gaussian blur
                intensity = intensity.unsqueeze(1)  # [B, 1, H, W]
                smoothed = F.conv2d(
                    intensity, 
                    gaussian_kernel, 
                    padding=kernel_size//2,
                    groups=1
                ).squeeze(1)  # [B, H, W]
            else:
                smoothed = intensity
            
            if self.gamma > 0:
                # Apply gamma correction
                prob_map = torch.pow(smoothed, self.gamma)
                # Add random noise
                prob_map = prob_map + torch.rand_like(prob_map) * 0.1
                # Normalize - add eps to avoid divide by zero
                eps = 1e-8
                prob_map = prob_map / (torch.sum(prob_map, dim=(1, 2), keepdim=True) + eps)
            else:
                prob_map = smoothed
        
        else:  # [C, H, W]
            intensity = rgb_image[0, :, :] * 0.2989 + rgb_image[1, :, :] * 0.5870 + rgb_image[2, :, :] * 0.1140  # [H, W]
            # norm to 0-1 by min-max
            intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
            if self.smooth_sigma > 0:

                kernel_size = int(2 * round(3 * self.smooth_sigma) + 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                

                coords = torch.arange(kernel_size, dtype=torch.float32, device=rgb_image.device) - (kernel_size - 1) / 2
                x, y = torch.meshgrid(coords, coords, indexing='ij')
                gaussian_kernel = torch.exp(-(x**2 + y**2) / (2 * self.smooth_sigma**2))
                gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
                gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
                
                intensity = intensity.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                smoothed = F.conv2d(
                    intensity, 
                    gaussian_kernel, 
                    padding=kernel_size//2,
                    groups=1
                ).squeeze(0).squeeze(0)  # [H, W]
            else:
                smoothed = intensity

            if self.gamma > 0:

                prob_map = torch.pow(smoothed, self.gamma)

                prob_map = prob_map + torch.rand_like(prob_map) * 0.1

                eps = 1e-8
                prob_map = prob_map / (torch.sum(prob_map) + eps)
            else:
                prob_map = smoothed
            
        return prob_map
    
    def sample_photons(self, rgb_image, num_photons):
        """Photon sampling processing
        
        Args:
            rgb_image: RGB image, shape [C, H, W]
            num_photons: number of photons to sample
            
        Returns:
            sampled_image: sampled image, shape [H, W]
        """
        if not isinstance(rgb_image, torch.Tensor):
            rgb_image = torch.from_numpy(rgb_image).float()
        
        prob_map = self.get_photon_probability_map(rgb_image)
        
        # norm with min-max to 0-1
        prob_map = (prob_map - prob_map.min()) / (prob_map.max() - prob_map.min())
        prob_sum = prob_map.sum()
        if prob_sum > 0:
            prob_map = prob_map / prob_sum  
        else:
            prob_map = torch.ones_like(prob_map) / prob_map.numel()
        
        rows, cols = prob_map.shape
        flat_prob = prob_map.reshape(-1)
        
        sampled_indices = torch.multinomial(flat_prob, num_photons, replacement=True)
        
        sampled_image = torch.zeros((rows, cols), dtype=torch.float32, device=rgb_image.device)
        
        y_coords = sampled_indices // cols
        x_coords = sampled_indices % cols
        
        sampled_image[y_coords, x_coords] = 1.0
        
        return sampled_image
    
    # def rgb_to_spike(self, rgb_batch):
    #     """Convert a batch of RGB images to spike stream data
        
    #     Args:
    #         rgb_batch: batch of RGB images, shape [B, C, H, W], values in [0, 1]
            
    #     Returns:
    #         spike_stream: spike stream data, shape [B, N, H, W], where N is number of sampling frames
    #     """
        
    #     def img_to_spike(img, gain_amp=0.5, v_th=1.0, n_timestep=8):
    #         # import pdb; pdb.set_trace()
    #         '''
    #         Pulse simulator: convert image to pulses

    #         :param img: image as numpy.ndarray size: h x w
    #         :param gain_amp: gain factor
    #         :param v_th: threshold
    #         :param n_timestep: number of timesteps
    #         :return: pulse data numpy.ndarray
    #         '''
    #         # img = img[0, :, :] * 0.2989 + img[1, :, :] * 0.5870 + img[2, :, :] * 0.1140  # [H, W]
    #         h, w = img.shape
            
    #         # img = (img - img.min()) / img.max()
    #         if img.max() > 1:
    #             img = img / 255.
    #         img = img - img.min()
    #         img = img / img.max()
    #         # if img.max() <= 1.0 and img.min() >= -1.0 and img.min() < 0.0:
    #         #     img = (img + 1) / 2.0
    #         # print(img.max(), img.min())
    #         assert img.max() <= 1.0 and img.min() >= 0.0
    #         mem = np.zeros_like(img)
    #         spks = np.zeros((n_timestep, h, w))
    #         for t in range(n_timestep):
    #             mem += img * gain_amp
    #             spk = (mem >= v_th)
    #             mem = mem * (1 - spk)
    #             spks[t, :, :] = spk
    #         return spks.astype(np.float32)
        
    #     if not isinstance(rgb_batch, torch.Tensor):
    #         rgb_batch = torch.from_numpy(rgb_batch).float()
        
    #     batch_size = rgb_batch.shape[0]
    #     height, width = rgb_batch.shape[2], rgb_batch.shape[3]
        
    #     spike_list = []
    #     device = rgb_batch.device
    #     for b in range(batch_size):
            
    #         rgb_img = rgb_batch[b].cpu().numpy()  # [3, H, W]
    #         gray_img = (
    #                     0.2989 * rgb_img[0] + 
    #                     0.5870 * rgb_img[1] + 
    #                     0.1140 * rgb_img[2]
    #                 ).astype(np.float32)
    #         spike_np = img_to_spike(gray_img, gain_amp=0.5, v_th=1.0, n_timestep=self.photon_samples)
    #         spike_tensor = torch.from_numpy(spike_np).float()  # [N, H, W]
    #         spike_list.append(spike_tensor)
        
        
        
    #     spike_stream = torch.stack(spike_list, dim=0)

    #     return spike_stream.to(device)
    
    def rgb_to_spike(self, rgb_batch):


        if not isinstance(rgb_batch, torch.Tensor):
            rgb_batch = torch.from_numpy(rgb_batch).float()
        
        batch_size = rgb_batch.shape[0]
        height, width = rgb_batch.shape[2], rgb_batch.shape[3]

        spike_stream = torch.zeros((batch_size, self.photon_samples, height, width), 
                                  dtype=torch.float32, device=rgb_batch.device)

        for b in range(batch_size):

            total_pixels = height * width
            photons_per_sample = int(total_pixels * self.target_coverage)
            

            for n in range(self.photon_samples):

                sampled_image = self.sample_photons(rgb_batch[b], photons_per_sample)
                spike_stream[b, n] = sampled_image
        

        spike_stream = (spike_stream > 0).float()
            
        return spike_stream

class OnlineProcessDataset(Dataset):

    
    def __init__(self, 
                 image_folder, 
                 transform=None, 
                 img_h=256,
                 img_w=256,
                 kernel_size=100,
                 blur_intensity=150,
                 blur_samples=10):

        self.image_folder = image_folder
        self.img_h = img_h
        self.img_w = img_w
        

        self.image_paths = self._get_image_paths()
        

        self.kernel_size = kernel_size
        self.blur_intensity = blur_intensity
        self.blur_samples = blur_samples

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_h, img_w), antialias=True), 
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
    
    def _get_image_paths(self):
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.JPEG', '.JPG', '.PNG')
        image_paths = []
        
        # Recursively search all subfolders
        for root, _, files in os.walk(self.image_folder):
            for file in files:
                if file.endswith(valid_extensions):
                    image_paths.append(os.path.join(root, file))
        
        if not image_paths:
            raise ValueError(f"No valid image files found in {self.image_folder} and its subfolders")
            
        print(f"Found {len(image_paths)} valid image files")  # Add log
        return sorted(image_paths)

    def create_motion_blur_kernel(self, kernel_size, angle, intensity=1.0):
        """Create motion blur kernel"""
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        angle_rad = np.deg2rad(angle)
        
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        
        for i in np.linspace(-center, center, kernel_size*2):
            x = center + dx * i
            y = center + dy * i
            
            x_int, y_int = int(round(x)), int(round(y))
            if 0 <= y_int < kernel_size and 0 <= x_int < kernel_size:
                distance = np.sqrt((x-center)**2 + (y-center)**2)
                weight = np.exp(-distance**2 / (2*(center*intensity)**2))
                kernel[y_int, x_int] += weight
        
        if np.sum(kernel) <= 1e-10:
            kernel[center, center] = 1.0
            
        return kernel / np.sum(kernel)

    def apply_motion_blur(self, image):
        """Apply motion blur effect"""
        original = image.copy().astype(np.float32)
        blurred = np.zeros_like(original, dtype=np.float32)
        
        main_angle = np.random.uniform(0, 360)
        
        for _ in range(self.blur_samples):
            angle_variance = 30
            current_angle = main_angle + np.random.uniform(-angle_variance, angle_variance)
            current_intensity = self.blur_intensity * (1 + np.random.uniform(-0.5, 0.5)) / 100.0
            
            kernel = self.create_motion_blur_kernel(
                self.kernel_size, 
                current_angle,
                current_intensity
            )
            
            sample = np.zeros_like(original, dtype=np.float32)
            for c in range(3):
                sample[:,:,c] = cv2.filter2D(
                    original[:,:,c], 
                    -1, 
                    kernel,
                    borderType=cv2.BORDER_REFLECT
                )
            
            weight = np.random.uniform(0.8, 1.2)
            blurred += sample * weight / self.blur_samples
        
        return np.clip(blurred, 0, 255).astype(np.uint8)

    def _process_image(self, image):
        """Process a single image"""
        try:
            # # Read the original image
            # image = cv2.imread(img_path)
            # if image is None:
            #     raise ValueError(f"Unable to read image: {img_path}")
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # # First resize and crop the image
            # h, w = image.shape[:2]
            # # Calculate the shorter side
            # shorter_side = min(h, w)
            # # Calculate crop start position
            # start_h = (h - shorter_side) // 2
            # start_w = (w - shorter_side) // 2
            # # Center crop
            # image = image[start_h:start_h+shorter_side, start_w:start_w+shorter_side]
            # # Resize to target size
            # image = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_LANCZOS4)
            # Apply motion blur
            blurred = self.apply_motion_blur(image)
            
            return (Image.fromarray(image), 
                   Image.fromarray(blurred))
            
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            return (Image.new('RGB', (self.img_h, self.img_w)),
                   Image.new('RGB', (self.img_h, self.img_w)))
    
    def __len__(self):
        """Return the number of images in the dataset"""
        return len(self.image_paths)
    
    # def __getitem__(self, idx):
    #     """Get one data item"""
    #     img_path = self.image_paths[idx]
    #     dir_path = os.path.dirname(os.path.dirname(img_path))
    #     file_name = img_path.split('/')[-1]
    #     image = cv2.imread(img_path)
    #     # blur_img = cv2.imread(f"/share/project/emllm_mnt.1d/hpfs/baaiei/daigaole/code/spikegen/task1/datasets/deblur_data/synthetic_gt/deblur_mix/{img_path.split('/')[-1]}")
    #     blur_img = cv2.imread(f"{dir_path}/deblur_mix/{file_name}")
    #     if image is None:
    #         raise ValueError(f"Unable to read image: {img_path}")
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
    #     # blur_img = self.apply_motion_blur(blur_img)
        
    #     image = Image.fromarray(image)
    #     blur_img = Image.fromarray(blur_img)
    #     if self.transform:
    #         rgb_img = self.transform(image)
    #         blur_img = self.transform(blur_img)
        
        
    #     return {
    #         'rgb': rgb_img,          # [3, H, W]
    #         'blur': blur_img,        # [3, H, W]
    #         'path': img_path,
    #     }
        
    def __getitem__(self, idx):
        """Get one data item"""
        img_path = self.image_paths[idx]

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Unable to read image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        blur_img = self.apply_motion_blur(image)
        
        image = Image.fromarray(image)
        blur_img = Image.fromarray(blur_img)
        if self.transform:
            rgb_img = self.transform(image)
            blur_img = self.transform(blur_img)
        
        
        return {
            'rgb': rgb_img,          # [3, H, W]
            'blur': blur_img,        # [3, H, W]
            'path': img_path,
        }



def get_online_process_dataloader(image_folder, batch_size=8, num_workers=4, 
                                img_h=256, img_w=256, shuffle=True, distributed=False,
                                rank=0, world_size=1, **kwargs):
    """Create dataloader for online processing dataset"""
    dataset = OnlineProcessDataset(
        image_folder=image_folder,
        img_h=img_h,
        img_w=img_w,
        **kwargs
    )
    
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
        shuffle = False
    else:
        sampler = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def visualize_images(batch, spike_stream, prob_map, batch_idx):
    """Visualize RGB image, blurred image, probability map, and spike stream data"""
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    
    # Create output directory
    output_dir = "visualization_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the first sample for visualization
    rgb_img = batch['rgb'][0]
    blur_img = batch['blur'][0]
    
    # Convert tensors to numpy arrays and reshape
    rgb_np = rgb_img.permute(1, 2, 0).cpu().numpy()
    blur_np = blur_img.permute(1, 2, 0).cpu().numpy()
    
    # Denormalize
    rgb_np = (rgb_np * 0.5 + 0.5).clip(0, 1)
    blur_np = (blur_np * 0.5 + 0.5).clip(0, 1)
    
    # Create image grid
    plt.figure(figsize=(20, 10))
    
    # Show RGB image
    plt.subplot(2, 4, 1)
    plt.imshow(rgb_np)
    plt.title('RGB Image')
    plt.axis('off')
    
    # Show blurred image
    plt.subplot(2, 4, 2)
    plt.imshow(blur_np)
    plt.title('Blur Image')
    plt.axis('off')
    
    # Show probability map
    plt.subplot(2, 4, 3)
    plt.imshow(prob_map, cmap='hot')
    plt.colorbar()
    plt.title('Photon Sampling Probability Map')
    plt.axis('off')
    
    # Randomly choose one spike frame as mask and visualize masked RGB image
    spike_frame = spike_stream[0, np.random.randint(0, spike_stream.shape[1])].cpu().numpy()
    # Fix: expand spike_frame to match shape of rgb_np
    spike_frame_expanded = np.expand_dims(spike_frame, axis=2)  # Add channel dimension
    masked_rgb = blur_np * spike_frame_expanded
    plt.subplot(2, 4, 4)
    plt.imshow(masked_rgb)
    plt.title('Masked RGB Image')
    plt.axis('off')
    
    
    
    # Show spike stream data (first 4 frames)
    for i in range(min(4, spike_stream.shape[1])):
        plt.subplot(2, 4, 5+i)
        spike_frame = spike_stream[0, i].cpu().numpy()
        plt.imshow(spike_frame, cmap='gray')
        plt.title(f'Spike Frame {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/visualization_batch_{batch_idx}.png")
    plt.close()
    
    # Save individual spike stream frames
    for i in range(spike_stream.shape[1]):
        plt.figure(figsize=(8, 8))
        spike_frame = spike_stream[0, i].cpu().numpy()
        plt.imshow(spike_frame, cmap='gray')
        plt.colorbar()
        plt.title(f'Spike Frame {i+1}')
        plt.axis('off')
        plt.savefig(f"{output_dir}/spike_frame_{batch_idx}_{i}.png")
        plt.close()
    
    print(f"Visualization results saved to {output_dir} directory")


# Usage example
if __name__ == "__main__":
    # Image folder path
    image_folder = "/path/to/SSIR_datasets"
    # "/path/to/data/imagenet1k/val"
    
    # Create dataset
    dataset = OnlineProcessDataset(image_folder)
    print(f"Dataset size: {len(dataset)}")
    
    # Get one sample and display
    sample = dataset[0]
    print(f"RGB image shape: {sample['rgb'].shape}")
    print(f"Blurred image shape: {sample['blur'].shape}")
    print(f"Image path: {sample['path']}")
    
    # Create dataloader
    dataloader = get_online_process_dataloader(
        image_folder,
        batch_size=4,
        kernel_size=50,
        blur_intensity=50,
        img_h=720,
        img_w=1280
    )
    
    # Create spike stream converter
    spike_converter = SpikeConverter(
        photon_samples=8,
        target_coverage=0.1,
        smooth_sigma=1,
        gamma=2.5
    )
    
    # Test dataloader and spike stream converter
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx+1}:")
        print(f"RGB image shape: {batch['rgb'].shape}")
        print(f"Blurred image shape: {batch['blur'].shape}")
        
        # Convert to spike stream data
        spike_stream = spike_converter.rgb_to_spike(batch['rgb'])
        print(f"Spike stream shape: {spike_stream.shape}")
        
        # Get probability map
        prob_map = spike_converter.get_photon_probability_map(batch['rgb'][0])
        print(f"Probability map shape: {prob_map.shape}")
        
        # Visualize images
        visualize_images(batch, spike_stream, prob_map, batch_idx)
        
        # Only test the first 2 batches
        if batch_idx >= 1:
            break
