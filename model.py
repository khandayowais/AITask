from torch import nn


class CNNModel(nn.Module):
    """
        CNN Model for FashionMNIST Dataset for Binary Classification
        2 Conv2d layers
        2 Fully Connected Layers
    """

    def __init__(self):
        super().__init__()  # super class constructor

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        '''
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        '''
        self.fc1 = nn.Linear(in_features=288, out_features=8)
        self.drop = nn.Dropout2d(0.6)
        self.fc2 = nn.Linear(in_features=8, out_features=2)
        #self.fc3 = nn.Linear(in_features=16, out_features=2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        #out = self.layer3(out)
        #print(out.shape)
        #out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        #out = self.fc3(out)

        return out
