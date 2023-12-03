'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class MnistLogistic(torch.nn.Module):
    def __init__(self):
        super(MnistLogistic, self).__init__()

        self.linear = nn.Linear(in_features=784, out_features=10)

    def forward(self, x, last=False):

        x = torch.flatten(x, 1)
        logits = self.linear(x)
        # outputs = F.softmax(logits, dim=1)

        return logits


class DogCatLogistic(torch.nn.Module):
    def __init__(self):
        super(DogCatLogistic, self).__init__()

        self.linear = nn.Linear(in_features=512, out_features=2)

    def forward(self, x):

        x = torch.flatten(x, 1)
        logits = self.linear(x)

        return logits


class DogCatMLP(torch.nn.Module):
    def __init__(self, factor=1):
        super(DogCatMLP, self).__init__()
        self.linear1 = nn.Linear(in_features=512, out_features=256)

        self.linear1.weight.data *= factor
        self.linear1.bias.data *= factor

        self.linear2 = nn.Linear(in_features=256, out_features=2)

        self.linear2.weight.data *= factor
        self.linear2.bias.data *= factor

        self.relu = nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.relu(x)
        logits = self.linear2(x)
        return logits




class MnistLeNet(torch.nn.Module):
    def __init__(self):
        super(MnistLeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding_mode='replicate')
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5, stride=1, padding_mode='replicate')

        self.linear1 = nn.Linear(in_features=640, out_features=500)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=500, out_features=10)

    def forward(self, x, last=False):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = torch.flatten(x, 1)
        x = self.relu(self.linear1(x))
        logits = self.linear2(x)
        # outputs = F.softmax(logits, dim=1)

        return logits

    def getFeature(self, x, numpy=True):
        if x.shape[1]==28:
          x = np.moveaxis(x, 3, 1)
        x = torch.Tensor(x).cuda()
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        if numpy:
          return x.detach().cpu().numpy()
        else:
          return x

class MnistLargeCNN(torch.nn.Module):
    def __init__(self):
        super(MnistLargeCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding_mode='replicate')
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5, stride=1, padding_mode='replicate')

        self.linear1 = nn.Linear(in_features=640, out_features=500)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=500, out_features=128)
        self.linear3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x, last=False):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = torch.flatten(x, 1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        logits = self.linear3(x)

        return logits

    def getFeature(self, x, numpy=True):
        if x.shape[1]==28:
          x = np.moveaxis(x, 3, 1)
        x = torch.Tensor(x).cuda()
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        if numpy:
          return x.detach().cpu().numpy()
        else:
          return x


class SmallCNN_CIFAR(nn.Module):
    def __init__(self):

        super(SmallCNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, last=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        if last:
          return output, x
        else:
          return output

    def getFeature(self, x, numpy=True):
        if x.shape[1]==32:
          x = np.moveaxis(x, 3, 1)
        x = torch.Tensor(x).cuda()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        if numpy:
          return x.detach().cpu().numpy()
        else:
          return x

    def get_embedding_dim(self):
        return 84



class BinaryMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=100):
        super(BinaryMLP, self).__init__()
        self.linear1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=2)
        #self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.relu(x)
        logits = self.linear2(x)
        return logits





