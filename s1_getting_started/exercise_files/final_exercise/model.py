from torch import nn


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

