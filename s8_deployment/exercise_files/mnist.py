from torch import nn, optim
import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=5, out_channels=3, kernel_size=3, stride=2)
                                )
        self.fc = nn.Sequential(nn.Linear(432, 100),
                                nn.ReLU(),
                                nn.Linear(100,10),
                                nn.LogSoftmax(dim=1)
                                )

    def forward(self, x):
        x = self.cnn(x).view(x.size(0), -1)
        return self.fc(x)

transform = transforms.Compose([transforms.ToTensor()])
dataset = MNIST("./mnist", download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=32)
model = MyAwesomeModel()
critirion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
losses = []
for e in range(25):
    running_loss = 0
    for images, labels in dataloader:
        out = model(images)
        loss = critirion(out, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
    print(running_loss)
    losses.append(running_loss)

script_model = torch.jit.script(model)
script_model.save('deployable_model.pt')