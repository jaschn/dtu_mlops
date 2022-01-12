"""
LFW dataloading
"""
import argparse
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import glob
import os
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

class LFWDataset(Dataset):
    def __init__(self, path_to_folder: str, transform) -> None:
        self.transform = transform
        folder = os.path.join(path_to_folder, "**/*.jpg")
        self.paths = glob.glob(folder)
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_to_folder', default='./lfw', type=str)
    parser.add_argument('-batch_size', default=512, type=int)
    parser.add_argument('-num_workers', default=1, type=int)
    parser.add_argument('-visualize_batch', default=False, action='store_true')
    parser.add_argument('-get_timing', default=True, action='store_true')
    parser.add_argument('-batches_to_check', default=2, type=int)
    
    args = parser.parse_args()
    
    lfw_trans = transforms.Compose([
        transforms.RandomAffine(5, (0.1, 0.1), (0.5, 2.0)),
        transforms.ToTensor()
    ])
    
    # Define dataset
    dataset = LFWDataset(args.path_to_folder, lfw_trans)
    
    # Define dataloader

    
    if args.visualize_batch:
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers
        )
        for images in dataloader:
            grid = make_grid(images)
            show(grid)
            plt.show()
            break

        
    if args.get_timing:
        timing_mean = []
        timing_std = []
        print(f"cpu count: {os.cpu_count()}")
        r = range(0, os.cpu_count(), 2)
        for worker in r:
            dataloader = DataLoader(
                dataset, 
                batch_size=args.batch_size, 
                shuffle=False,
                num_workers=worker
            )
            # lets do some repetitions
            res = []
            for _ in range(5):
                start = time.time()
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx > args.batches_to_check:
                        break
                end = time.time()

                res.append(end - start)
                
            res = np.array(res)
            print(f'Timing with {worker} workers: {np.mean(res)}+-{np.std(res)}')
            timing_mean.append(np.mean(res))
            timing_std.append(np.std(res))
        plt.errorbar(r, timing_mean, yerr=timing_std)
        plt.show()
