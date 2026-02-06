
import xlrd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
def calculate_accuracy(x1, x2):
    x1=(x1>0.5).astype(int)
    x2=(x2>0.5).astype(int)
    acc = np.mean(np.abs(x1-x2))
    return acc
    #return r_squared(x1[0], x2[0]),r_squared(x1[1], x2[1])
def load_data(filename):
    book = xlrd.open_workbook(filename)
    sheet1 = book.sheet_by_name('Sheet1')
    sheet2 = book.sheet_by_name('Sheet2')
    m, n1, n2= sheet1.nrows, sheet1.ncols, sheet2.ncols
    lvs = np.zeros((m, n1))
    samples = np.zeros((m, n2))
    for i in range(m):
        for j in range(n1):
            lvs[i, j] = sheet1.cell(i, j).value
        for j in range(n2):
            samples[i, j] = 1 if sheet2.cell(i, j).value > 0.5 else 0
    tr_lvs, va_lvs, te_lvs = lvs[:int(m * 0.90)], lvs[int(m * 0.90):int(m * 0.95)], lvs[int(m * 0.95):m]
    tr_samples, va_samples, te_samples = samples[:int(m * 0.90)], samples[int(m * 0.90):int(m * 0.95)], samples[int(m * 0.95):m]
    return tr_lvs, va_lvs, te_lvs, tr_samples, va_samples, te_samples,






class CustomDecoder(nn.Module):
    def __init__(self, input_dim):
        super(CustomDecoder, self).__init__()

        # Fully connected layer
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 1024)
        self.bn2 = nn.BatchNorm1d(1024)

        # Define the number of output channels
        self.output_channels = 128
        self.fc3 = nn.Linear(1024, 5 * 5 * self.output_channels)
        self.bn3 = nn.BatchNorm1d(5 * 5 * self.output_channels)

        # Transposed convolution layers
        self.deconv1 = nn.ConvTranspose2d(self.output_channels, 64, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(64)

        self.deconv2 = nn.ConvTranspose2d(64, 48, kernel_size=3, stride=2)
        self.bn5 = nn.BatchNorm2d(48)

        self.deconv3 = nn.ConvTranspose2d(48, 32, kernel_size=3, stride=2)
        self.bn6 = nn.BatchNorm2d(32)

        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1)
        self.bn7 = nn.BatchNorm2d(16)

        self.deconv5 = nn.ConvTranspose2d(16, 1, kernel_size=2, stride=1)

    def forward(self, x):
        # Flattening and fully connected layers
        x = x.view(x.size(0), -1)   # Flattening
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))

        # Here we assume the output of the last layer will be used as input for the feature map
        x = F.relu(self.bn3(self.fc3(x)))
        x = x.view(-1, self.output_channels, 5, 5)    # Reshape into a 5x5 feature map


        # 转置卷积层
        x = F.relu(self.bn4(self.deconv1(x)))  # becomes (batch_size, 64, 11, 11)
        x = F.relu(self.bn5(self.deconv2(x)))  # becomes (batch_size, 48, 23, 23)
        x = F.relu(self.bn6(self.deconv3(x)))  # becomes (batch_size, 32, 47, 47)
        x = F.relu(self.bn7(self.deconv4(x)))  # becomes (batch_size, 16, 49, 49)

        x = self.deconv5(x)  # becomes (batch_size, 1, 50, 50)

        # Additional processing for rotation and flipping
        x1 = x  # Original output
        x2 = x1.rot90(1, dims=[2, 3])  # Rotate 90 degrees
        x3 = x2.rot90(1, dims=[2, 3])  # Rotate another 90 degrees
        x4 = x3.rot90(1, dims=[2, 3])  # Rotate another 90 degrees
###########################################
        x5 = x1.flip(dims=[3])  # Horizontal flip
        x6 = x2.flip(dims=[2])  # Vertical flip
        x7 = x3.flip(dims=[3])  # Horizontal flip
        x8 = x4.flip(dims=[2])  # Vertical flip

        # Sum all versions and calculate the average
        x = (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8) / 8.0
        x = torch.sigmoid(x)  # Use Sigmoid activation function to ensure output is between 0 and 1

        return x
