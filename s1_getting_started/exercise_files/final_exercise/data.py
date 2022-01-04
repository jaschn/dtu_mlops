import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

class CorruptMNISTset(Dataset):
    def __init__(self, type="train"):
        base_path = "./../../../data/corruptmnist"
        if type == "train":
            with np.load("./../../../data/corruptmnist/train_0.npz") as f:
                im = torch.from_numpy(f["images"]).type(torch.float)
                self.images = im.view(im.size(0), 1, im.size(1), im.size(2))
                self.labels = torch.from_numpy(f["labels"]).type(torch.long)
            for i in range(1, 5):
                with np.load(os.path.join(base_path, f"train_{i}.npz")) as f:
                    im = torch.from_numpy(f["images"]).type(torch.float)
                    im = im.view(im.size(0), 1, im.size(1), im.size(2))
                    self.images = torch.cat([self.images, im], dim=0)        
                    labels = torch.from_numpy(f["labels"]).type(torch.long)
                    self.labels = torch.cat([self.labels, labels], dim=0)
        elif type == "test":
            with np.load("./../../../data/corruptmnist/test.npz") as f:
                im = torch.from_numpy(f["images"]).type(torch.float)
                self.images = im.view(im.size(0), 1, im.size(1), im.size(2))                
                self.labels = torch.from_numpy(f["labels"]).type(torch.long)
        else:
            assert "wrong option"

        assert len(self.images) == len(self.labels), "image count does not match label count"


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

def mnist():
    trainset = CorruptMNISTset(type="train")
    testset = CorruptMNISTset(type="test")
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8)
    testloader = DataLoader(testset, batch_size=64, shuffle=True, num_workers=8)
    return trainloader, testloader 
