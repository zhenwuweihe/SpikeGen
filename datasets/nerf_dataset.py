import os 
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import glob
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

class NerfProcessedDataset(Dataset):
    """Processed NeRF synthetic dataset loader
    
    Loads processed blur images and spike npy files as input pairs, using original RGB images as ground truth
    """
    
    def __init__(self, root_dir, scene_list=None, transform=None, img_size=256):
        """
        Initialize dataset
        
        Args:
            root_dir (str): Root directory of the dataset, containing blur, spike, and rgb subfolders
            scene_list (list): List of scenes to include, if None all scenes are included
            transform (callable, optional): Optional transform to be applied to images
            img_size (int): Output image size
        """
        self.root_dir = root_dir
        self.img_size = img_size
        
        # Get all scenes
        all_scenes = self._get_scenes()
        
        # If a specific list of scenes is provided, use only those
        if scene_list is not None:
            self.scenes = [scene for scene in all_scenes if scene in scene_list]
        else:
            self.scenes = all_scenes
            
        # Generate list of (blur, spike, rgb) image paths
        self.image_pairs = self._generate_image_pairs()
        
        # Set image transformation
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1,1]
        ])
    
    def _get_scenes(self):
        """Get all scenes in the dataset"""
        # Check all subdirectories under the blur folder
        blur_path = os.path.join(self.root_dir, 'blur')
        if not os.path.exists(blur_path):
            raise ValueError(f"Blur folder does not exist: {blur_path}")
            
        scenes = [d for d in os.listdir(blur_path) 
                 if os.path.isdir(os.path.join(blur_path, d))]
        
        if not scenes:
            raise ValueError(f"No scenes found in {blur_path}")
            
        return scenes
    
    def _generate_image_pairs(self):
        """Generate list of (blur, spike, rgb) image paths"""
        image_pairs = []
        
        for scene in self.scenes:
            # Get blur image paths
            blur_dir = os.path.join(self.root_dir, 'blur', scene)
            blur_imgs = sorted(glob.glob(os.path.join(blur_dir, '*.png')))
            
            # Get spike npy file paths
            spike_dir = os.path.join(self.root_dir, 'spike', scene)
            spike_files = sorted(glob.glob(os.path.join(spike_dir, '*.npy')))
            
            # Get rgb image paths (ground truth)
            rgb_dir = os.path.join(self.root_dir, 'rgb', scene)
            rgb_imgs = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
            
            # Check file count consistency
            if len(blur_imgs) != len(spike_files) or len(blur_imgs) != len(rgb_imgs):
                print(f"Warning: File count mismatch in scene {scene} - blur: {len(blur_imgs)}, spike: {len(spike_files)}, rgb: {len(rgb_imgs)}")
                # Use the minimum length
                min_len = min(len(blur_imgs), len(spike_files), len(rgb_imgs))
                blur_imgs = blur_imgs[:min_len]
                spike_files = spike_files[:min_len]
                rgb_imgs = rgb_imgs[:min_len]
            
            # Add to image pair list
            for blur_path, spike_path, rgb_path in zip(blur_imgs, spike_files, rgb_imgs):
                # Ensure filenames match
                blur_basename = os.path.basename(blur_path)
                spike_basename = os.path.basename(spike_path)
                rgb_basename = os.path.basename(rgb_path)
                
                if blur_basename.split('.')[0] == spike_basename.split('.')[0] == rgb_basename.split('.')[0]:
                    image_pairs.append({
                        'scene': scene,
                        'blur': blur_path,
                        'spike': spike_path,
                        'rgb': rgb_path
                    })
                else:
                    print(f"Warning: Filename mismatch - blur: {blur_basename}, spike: {spike_basename}, rgb: {rgb_basename}")
        
        if not image_pairs:
            raise ValueError(f"No valid image pairs found")
            
        return image_pairs
    
    def _load_image(self, path):
        """Load and preprocess image
        
        Args:
            path (str): Image path
            
        Returns:
            torch.Tensor: Processed image tensor
        """
        try:
            img = Image.open(path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
                
            return img
            
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Return a blank image
            return torch.zeros(3, self.img_size, self.img_size)
    
    def _load_spike(self, path):
        """Load and preprocess spike stream
        
        Args:
            path (str): Path to npy file containing multi-frame spike data
            
        Returns:
            torch.Tensor: Processed spike data tensor [frames, height, width]
        """
        try:
            # Load npy file - expected shape [num_frames, height, width]
            spike_stream = np.load(path)
            
            # Ensure data is in [0,1] range
            if spike_stream.max() > 1.0:
                spike_stream = spike_stream / 255.0
            
            # Get number of frames
            num_frames = spike_stream.shape[0]
            
            # Resize each frame and convert to tensor
            frames_list = []
            for i in range(num_frames):
                frame = spike_stream[i]
                frame_tensor = torch.from_numpy(frame).float()
                
                # Resize
                if frame.shape[0] != self.img_size or frame.shape[1] != self.img_size:
                    frame_tensor = transforms.functional.resize(
                        frame_tensor.unsqueeze(0),  # Add channel dimension
                        (self.img_size, self.img_size),
                        interpolation=transforms.InterpolationMode.NEAREST
                    ).squeeze(0)  # Remove channel dimension
                
                frames_list.append(frame_tensor)
            
            # Stack all frames
            if frames_list:
                spike_tensor = torch.stack(frames_list)  # [num_frames, H, W]
            else:
                # If no frames, create a blank tensor
                spike_tensor = torch.zeros(1, self.img_size, self.img_size)
            
            return spike_tensor
            
        except Exception as e:
            print(f"Error loading spike data {path}: {e}")
            # Return a blank spike tensor
            return torch.zeros(1, self.img_size, self.img_size)
    
    def __len__(self):
        """Return number of image pairs in dataset"""
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        """Get an image pair
        
        Args:
            idx (int): Index
            
        Returns:
            dict: Dictionary containing blur, spike, and rgb images
        """
        item = self.image_pairs[idx]
        
        # Load images and spike data
        blur_img = self._load_image(item['blur'])
        spike_data = self._load_spike(item['spike'])  # Now [num_frames, H, W]
        rgb_img = self._load_image(item['rgb'])
        
        return {
            'scene': item['scene'],
            'blur': blur_img,         # [3, H, W]
            'spike': spike_data,      # [num_frames, H, W]
            'rgb': rgb_img,           # [3, H, W]
            'blur_path': item['blur'],
            'spike_path': item['spike'],
            'rgb_path': item['rgb']
        }
    
    def visualize_item(self, idx):
        """Visualize image pair at specified index
        
        Args:
            idx (int): Index
        """
        item = self[idx]
        
        # Get number of spike frames
        num_frames = item['spike'].shape[0]
        
        # Show up to 4 spike frames
        num_display_frames = min(4, num_frames)
        
        plt.figure(figsize=(15, 5 + 5 * (num_display_frames > 0)))
        
        # Show blurred image
        plt.subplot(2, 2, 1)
        blur_img = item['blur'].permute(1, 2, 0).numpy()
        plt.imshow(np.clip(blur_img, 0, 1))
        plt.title(f'Blurred Image\n{item["scene"]}')
        plt.axis('off')
        
        # Show RGB ground truth
        plt.subplot(2, 2, 2)
        rgb_img = item['rgb'].permute(1, 2, 0).numpy()
        plt.imshow(np.clip(rgb_img, 0, 1))
        plt.title('RGB Ground Truth')
        plt.axis('off')
        
        # Show multiple spike frames
        for i in range(num_display_frames):
            plt.subplot(2 + num_display_frames//2, 2, i + 3)
            spike_frame = item['spike'][i].numpy()
            plt.imshow(np.clip(spike_frame, 0, 1), cmap='gray')
            plt.title(f'Spike Frame #{i+1}/{num_frames}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print path info
        print(f"Scene: {item['scene']}")
        print(f"Blurred image: {item['blur_path']}")
        print(f"Spike data: {item['spike_path']} ({num_frames} frames)")
        print(f"RGB image: {item['rgb_path']}")


def get_nerf_dataloader(root_dir, batch_size=8, num_workers=4, scene_list=None, 
                        img_size=256, shuffle=True, distributed=False, rank=0, world_size=1):
    """Create NeRF synthetic dataset dataloader
    
    Args:
        root_dir (str): Root directory of the dataset
        batch_size (int): Batch size
        num_workers (int): Number of data loading workers
        scene_list (list): List of scenes to include, if None include all
        img_size (int): Output image size
        shuffle (bool): Whether to shuffle the data
        distributed (bool): Whether in distributed training
        rank (int): Rank of current process
        world_size (int): Total number of processes
        
    Returns:
        torch.utils.data.DataLoader: DataLoader object
    """
    # Create dataset
    dataset = NerfProcessedDataset(
        root_dir=root_dir,
        scene_list=scene_list,
        img_size=img_size
    )
    
    # Choose sampler based on distributed setting
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, 
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
        shuffle = False  # When using distributed sampler, disable shuffle in DataLoader
    else:
        sampler = None  # Let DataLoader handle sampler
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


# Usage example
if __name__ == "__main__":
    # Root directory of dataset
    root_dir = "path/to/processed_nerf_synthetic"
    
    # Create dataset
    dataset = NerfProcessedDataset(root_dir)
    print(f"Dataset size: {len(dataset)}")
    
    # Visualize a few samples
    for i in range(min(3, len(dataset))):
        dataset.visualize_item(i)
    
    # Create DataLoader
    dataloader = get_nerf_dataloader(root_dir, batch_size=4)
    
    # Test DataLoader
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx+1}:")
        print(f"Blurred image shape: {batch['blur'].shape}")
        print(f"Spike data shape: {batch['spike'].shape}")
        print(f"RGB image shape: {batch['rgb'].shape}")
        print(f"Scene: {batch['scene']}")
        
        # Only test first 2 batches
        if batch_idx >= 1:
            break
