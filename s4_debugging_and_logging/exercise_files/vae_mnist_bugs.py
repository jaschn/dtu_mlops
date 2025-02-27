"""
Adapted from
https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb

A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""
import os

import torch
import torchvision.transforms as transforms
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from Model import Model, Encoder, Decoder, TestCallback

def main():
    # Model Hyperparameters
    dataset_path = 'datasets'
    cuda = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if cuda else "cpu")
    batch_size = 100
    x_dim  = 784
    hidden_dim = 400
    latent_dim = 20
    lr = 1e-3


    # Data loading
    mnist_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
    test_dataset  = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)
    cpu_cnt = os.cpu_count()
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_cnt)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, num_workers=cpu_cnt)

        
    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)

    model = Model(Encoder=encoder, Decoder=decoder, lr=lr)

    trainer = Trainer(max_epochs=5, callbacks=[TestCallback()])
    trainer.fit(model, train_dataloaders=train_loader)

    # Generate reconstructions
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)      
            x_hat, _, _ = model(x)       
            break

    save_image(x.view(batch_size, 1, 28, 28), 'orig_data.png')
    save_image(x_hat.view(batch_size, 1, 28, 28), 'reconstructions.png')

    # Generate samples
    with torch.no_grad():
        noise = torch.randn(batch_size, latent_dim).to(DEVICE)
        generated_images = decoder(noise)
        
    save_image(generated_images.view(batch_size, 1, 28, 28), 'generated_sample.png')

if __name__ == "__main__":
    main()
