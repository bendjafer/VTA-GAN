import os
import cv2
import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
from lib.data.datasets import FD, find_classes, IMG_EXTENSIONS


def make_snippet_dataset(dir, class_to_idx):
    """
    Create dataset of video snippets instead of individual images
    Each snippet folder contains 16 frames
    """
    snippets = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            # Check if this directory contains .tif files (is a snippet folder)
            tif_files = [f for f in fnames if f.lower().endswith(('.tif', '.tiff'))]
            if len(tif_files) > 0:
                # This is a snippet folder
                snippet_path = root
                snippets.append((snippet_path, class_to_idx[target]))

    return snippets


class VideoSnippetDataset(data.Dataset):
    """
    Dataset for loading video snippets (folders containing 16 frames each)
    Each sample is a video snippet with 16 frames
    """
    
    def __init__(self, root, transform=None, num_frames=16):
        classes, class_to_idx = find_classes(root)
        snippets = make_snippet_dataset(root, class_to_idx)
        
        if len(snippets) == 0:
            raise(RuntimeError("Found 0 snippet folders in subfolders of: " + root + "\n"
                               "Expected folders containing .tif or .tiff files"))

        self.root = root
        self.snippets = snippets
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.num_frames = num_frames

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (video_snippet, target) where video_snippet is a tensor of shape 
                   (num_frames, 6, height, width) - 6 channels for lap and res components
        """
        snippet_path, target = self.snippets[index]
        
        # Get all .tif files in the snippet folder
        tif_files = sorted([f for f in os.listdir(snippet_path) 
                           if f.lower().endswith(('.tif', '.tiff'))])
        
        # Ensure we have exactly num_frames files
        if len(tif_files) < self.num_frames:
            # Pad with last frame if not enough frames
            while len(tif_files) < self.num_frames:
                tif_files.append(tif_files[-1])
        elif len(tif_files) > self.num_frames:
            # Take first num_frames if too many
            tif_files = tif_files[:self.num_frames]
        
        # Load and process all frames
        lap_frames = []
        res_frames = []
        
        for tif_file in tif_files:
            frame_path = os.path.join(snippet_path, tif_file)
            
            # Load frame
            img = cv2.imread(frame_path)
            if img is None:
                raise ValueError(f"Could not load frame: {frame_path}")
            
            # Apply frequency decomposition
            lap, res = FD(img)
            
            # Apply transforms
            if self.transform is not None:
                lap = self.transform(lap)
                res = self.transform(res)
            
            lap_frames.append(lap)
            res_frames.append(res)
        
        # Stack frames into tensors
        lap_tensor = torch.stack(lap_frames)  # (num_frames, 3, H, W)
        res_tensor = torch.stack(res_frames)  # (num_frames, 3, H, W)
        
        return lap_tensor, res_tensor, target

    def __len__(self):
        return len(self.snippets)


class VideoSnippetDatasetAug(VideoSnippetDataset):
    """
    Video snippet dataset with data augmentation
    """
    
    def __init__(self, root, transform=None, transform_aug=None, num_frames=16):
        super(VideoSnippetDatasetAug, self).__init__(root, transform, num_frames)
        self.transform_aug = transform_aug

    def __getitem__(self, index):
        """
        Returns:
            tuple: (lap_tensor, res_tensor, aug_tensor, target)
        """
        snippet_path, target = self.snippets[index]
        
        # Get all .tif files in the snippet folder
        tif_files = sorted([f for f in os.listdir(snippet_path) 
                           if f.lower().endswith(('.tif', '.tiff'))])
        
        # Ensure we have exactly num_frames files
        if len(tif_files) < self.num_frames:
            while len(tif_files) < self.num_frames:
                tif_files.append(tif_files[-1])
        elif len(tif_files) > self.num_frames:
            tif_files = tif_files[:self.num_frames]
        
        # Load and process all frames
        lap_frames = []
        res_frames = []
        aug_frames = []
        
        for tif_file in tif_files:
            frame_path = os.path.join(snippet_path, tif_file)
            
            # Load frame
            img = cv2.imread(frame_path)
            if img is None:
                raise ValueError(f"Could not load frame: {frame_path}")
            
            # Apply frequency decomposition
            lap, res = FD(img)
            
            # Apply normal transforms
            if self.transform is not None:
                lap = self.transform(lap)
                res = self.transform(res)
            
            # Create augmented version
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if self.transform_aug is not None:
                aug_img = self.transform_aug(img_pil)
            else:
                aug_img = self.transform(img_pil) if self.transform else img_pil
            
            lap_frames.append(lap)
            res_frames.append(res)
            aug_frames.append(aug_img)
        
        # Stack frames into tensors
        lap_tensor = torch.stack(lap_frames)  # (num_frames, 3, H, W)
        res_tensor = torch.stack(res_frames)  # (num_frames, 3, H, W)
        aug_tensor = torch.stack(aug_frames)  # (num_frames, 3, H, W)
        
        return lap_tensor, res_tensor, aug_tensor, target
