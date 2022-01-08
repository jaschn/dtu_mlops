import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam

class Encoder(nn.Module):  
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        self.training = True
        
    def forward(self, x):
        h_       = torch.relu(self.FC_input(x))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     

        std      = torch.exp(0.5*log_var)                         
        z        = self.reparameterization(mean, std)
        
        return z, mean, log_var
       
    def reparameterization(self, mean, std):
        epsilon = torch.rand_like(std)
        
        z = mean + std*epsilon
        
        return z
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h     = torch.relu(self.FC_hidden(x))
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat
    
def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD
    
class Model(pl.LightningModule):
    def __init__(self, Encoder, Decoder, lr):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.loss_function = loss_function
        self.lr = lr
                
    def forward(self, x):
        z, mean, log_var = self.Encoder(x)
        x_hat            = self.Decoder(z)
        
        return x_hat, mean, log_var

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)

        x_hat, mean, log_var = self(x)
        loss = self.loss_function(x, x_hat, mean, log_var)
        
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat, _, _ = self(x)    
        return x, x_hat  

class TestCallback(pl.Callback):
    def on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        print(f"Epoch: {trainer.current_epoch}")
