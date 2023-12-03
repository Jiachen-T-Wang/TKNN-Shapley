import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms

# general
import pandas as pd 
import numpy as np 
import copy
import pickle
import sys
import time
import os
import random

from helper import *
from utility_func import *
from prepare_data import *

# sbatch sample_mnist.sh Banzhaf_GT Logistic 5 5000 0.11 8 1

import argparse


x_train, y_train, x_test, y_test = get_dogcat()

model_type = 'ResNet18'

_, net = torch_dogcat_data_to_acc(model_type, x_train, y_train, x_test, y_test, weights=None, verbose=1, batch_size=32, lr=0.001, return_net=True)

torch.save(net.state_dict(), 'result/dogcat_extractor.net')



