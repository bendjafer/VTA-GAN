from options import Options
from lib.data.dataloader import load_video_data_FD_aug
from lib.models import load_model
import numpy as np
import prettytable as pt

def train_video(opt, dataset_name):
    """Train video model on UCSD2 dataset"""
    data = load_video_data_FD_aug(opt, dataset_name)
    model = load_model(opt, data, dataset_name)
    auc = model.train()
    return auc

def main():
    """ Training for video model
    """
    # Set up for UCSD2 dataset
    dataset_name = "ucsd2"
    
    opt = Options().parse()
    # Override model to use video version
    opt.model = 'ocr_gan_video'
    opt.dataset = dataset_name
    opt.num_frames = 8  # Number of frames per snippet
    
    print(f"Training OCR-GAN Video on {dataset_name}")
    auc = train_video(opt, dataset_name)
    
    print(f"Training completed. AUC: {auc:.4f}")

if __name__ == '__main__':
    main()
