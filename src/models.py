import torch.nn.functional as F
from torch import nn


class CustomNet(nn.Module):
    """
    Define custom model below
    """

    def __init__(self):
        super().__init__()


class MnistCNN(nn.Module):
    """
    CNN model for MNIST dataset
    Size:
    Class: 10
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.pool(self.conv3(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CifarCNN(nn.Module):
    """
    CNN model for CIFAR-10 dataset
    Size: 3@32x32
    Class: 10
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
