import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, n_z):
        super(Encoder, self).__init__()
        # 1th convolutional layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)

        # 2th convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn_pool1 = nn.BatchNorm2d(32)

        # 3th convolutional layer
        self.conv3 = nn.Conv2d(32, 48, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(48)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn_pool2 = nn.BatchNorm2d(48)

        # 4th convolutional layer
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn_pool3 = nn.BatchNorm2d(64)

        # 5th convolutional layer
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn_pool4 = nn.BatchNorm2d(128)

        # Fully connected layers
        self.fc1 = nn.Linear(128 , 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn_fc3 = nn.BatchNorm1d(128)

        #  Linear transformation for mean and log variance
        self.w1 = nn.Parameter(torch.randn(128, n_z) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(1))
        self.w2 = nn.Parameter(torch.randn(128, n_z) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # First convolutional layer
        x = F.relu(self.bn1(self.conv1(x)))

        # 2th convolutional layer
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn_pool1(x))

        # 3th convolutional layer
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = F.relu(self.bn_pool2(x))

        # 4th convolutional layer
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool3(x)
        x = F.relu(self.bn_pool3(x))

        # 5th convolutional layer
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool4(x)
        x = F.relu(self.bn_pool4(x))

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # Fully connected layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.bn_fc3(self.fc3(x))

        # Output mean and log variance
        z_mean = torch.matmul(x, self.w1) + self.b1
        z_log_var = torch.matmul(x, self.w2) + self.b2

        return z_mean, z_log_var



