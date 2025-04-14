import os
import glob
from typing import Callable, List, Tuple, Dict, Optional, Union

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from collections import namedtuple

from torchvision.transforms.functional import Tensor

DataSample = namedtuple('DataSample', ['color_path', 'mask_path', 'object_name', 'object_idx', 'location', 'scene'])

class AmazonPickingChallengeDataset(data.Dataset):
    """Dataset for Amazon Picking Challenge data.

    This dataset handles the large amount of data (130GB) by lazy loading images.
    Images are loaded only when requested, reducing memory usage.

    """

    def __init__(self,
                 root_dir: str,
                 locations: List[str] = ['shelf', 'tote'],
                 transform: Callable[[Image.Image], Tensor] = transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
                 ]),
    ):
        """Initialize the dataset.

        Args:
            root_dir (str): Root directory of the dataset (e.g., 'data/training')
            locations (List[str], optional): List of locations to use, e.g., ['shelf', 'tote']
            transform (callable, optional): Optional transform to be applied on input images
        """
        self.root_dir = root_dir
        self.locations = locations
        self.transform: Callable[[Image.Image], Tensor] = transform

        # Get all object classes
        self.object_classes = self._get_object_classes()
        self.num_classes = len(self.object_classes)

        # Create object to index mapping for one-hot encoding
        self.object_to_idx = {obj: idx for idx, obj in enumerate(self.object_classes)}

        # Get all image paths and their metadata
        self.samples = self._get_samples()

        print(f"Dataset initialized with {len(self.samples)} samples")
        print(f"Number of object classes: {self.num_classes}")

    def _get_object_classes(self) -> List[str]:
        """Get a sorted list of all object classes from directory names."""
        objects = set()
        for location in self.locations:
            location_path = os.path.join(self.root_dir, location)
            if not os.path.exists(location_path):
                continue
            for obj_dir in os.listdir(location_path):
                if os.path.isdir(os.path.join(location_path, obj_dir)):
                    objects.add(obj_dir)
        return sorted(list(objects))

    def _get_samples(self) -> List[DataSample]:
        """Create a list of all samples with their metadata."""
        samples = []
        for location in self.locations:
            location_path = os.path.join(self.root_dir, location)
            if not os.path.exists(location_path):
                continue

            for obj_name in os.listdir(location_path):
                obj_path = os.path.join(location_path, obj_name)
                if not os.path.isdir(obj_path) or obj_name not in self.object_classes:
                    raise ValueError(f"Invalid object: {obj_name}")

                for scene_dir in os.listdir(obj_path):
                    if not scene_dir.startswith('scene-'):
                        continue

                    scene_path = os.path.join(obj_path, scene_dir)
                    if not os.path.isdir(scene_path):
                        raise ValueError(f"Invalid scene: {scene_dir}")

                    # Find all color images
                    color_pattern = os.path.join(scene_path, 'frame-*.color.png')
                    for color_path in glob.glob(color_pattern):
                        frame_name = os.path.basename(color_path)  # e.g., "frame-000000.color.png"
                        frame_id = frame_name.split('.')[0]  # e.g., "frame-000000"

                        # Construct the mask path
                        mask_name = f"{frame_id}.mask.png"
                        mask_path = os.path.join(scene_path, 'masks', mask_name)

                        if not os.path.exists(mask_path):
                            raise ValueError(f"Missing mask for frame {frame_id} in scene {scene_dir}")

                        samples.append(DataSample(
                            color_path=color_path,
                            mask_path=mask_path,
                            object_name=obj_name,
                            object_idx=self.object_to_idx[obj_name],
                            location=location,
                            scene=scene_dir
                        ))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: Union[int, Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.

        Returns:
            tuple: (image, target) where image is a tensor of shape [C, H, W]
                  and target is a tensor of shape [H, W] with object category index
        """
        if torch.is_tensor(idx):
            idx = idx.long()

        sample = self.samples[idx]

        # Load color image
        img = Image.open(sample.color_path).convert('RGB')
        img_size = img.size
        img = self.transform(img)

        # Load mask image
        mask = Image.open(sample.mask_path).convert('L')
        mask_size = mask.size
        mask = np.array(mask)

        assert img_size == mask_size, f"Image size {img_size} does not match mask size {mask_size}"

        # Create one-hot encoding for the mask
        # If the mask has a positive value, it's the object
        target = torch.zeros(mask.shape[0], mask.shape[1], dtype=torch.long)

        object_pixels = mask.nonzero()
        target[object_pixels[0], object_pixels[1]] = sample.object_idx

        return img, target


def get_data_loader(root_dir: str = 'data/',
                    batch_size: int = 8,
                    num_workers: int = 4,
                    shuffle: bool = True,
                    locations: List[str] = ['shelf', 'tote']) -> DataLoader:
    """Create a data loader for the Amazon Picking Challenge dataset.

    Args:
        root_dir (str): Root directory of the dataset
        batch_size (int): Batch size for the data loader
        num_workers (int): Number of workers for prefetching
        shuffle (bool): Whether to shuffle the dataset
        locations (List[str], optional): List of locations to use
        target_size (Tuple[int, int]): Size to which images should be resized

    Returns:
        DataLoader: PyTorch data loader for the dataset
    """
    dataset = AmazonPickingChallengeDataset(
        root_dir=root_dir,
        locations=locations,
    )
    assert len(dataset) > 0

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Speeds up CPU to GPU transfer
        prefetch_factor=2  # Prefetch 2 batches per worker
    )

    return data_loader


def get_object_classes(root_dir: str = 'data/training',
                      locations: List[str] = ['shelf', 'tote']) -> List[str]:
    """Get a list of all object classes in the dataset.

    Args:
        root_dir (str): Root directory of the dataset
        locations (List[str], optional): List of locations to use

    Returns:
        List[str]: List of object class names
    """
    temp_dataset = AmazonPickingChallengeDataset(
        root_dir=root_dir,
        locations=locations
    )
    return temp_dataset.object_classes


# Example usage
if __name__ == "__main__":
    # Create a data loader
    loader = get_data_loader(batch_size=1, num_workers=1)

    # Get a batch of data
    for batch_idx, (imgs, masks) in enumerate(loader):
        print(f"Batch {batch_idx}:")
        print(f"  Images shape: {imgs.shape}")  # Should be [batch_size, 3, height, width]
        print(f"  Masks shape: {masks.shape}")  # Should be [batch_size, height, width, num_classes]

        # Only process one batch for this example
        break
