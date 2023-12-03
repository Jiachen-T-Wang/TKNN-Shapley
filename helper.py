import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import time
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
from os.path import exists
import warnings

from tqdm import tqdm

import scipy
from scipy.special import beta, comb
from random import randint
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from helper_privacy import PrivateKNN_mech, PrivateKNN_SV_RJ_mech
import prv_accountant
from prv_accountant.other_accountants import RDP
from prv_accountant import PRVAccountant, PoissonSubsampledGaussianMechanism
from prv_accountant.dpsgd import find_noise_multiplier
from sklearn.metrics.pairwise import cosine_similarity

save_dir = 'result/'



def kmeans_aucroc(value_array, cluster=False):

  n_data = len(value_array)

  # if cluster:
  #   X = value_array.reshape(-1, 1)
  #   kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
  #   min_cluster = min(kmeans.cluster_centers_.reshape(-1))
  #   pred = np.zeros(n_data)
  #   pred[value_array < min_cluster] = 1
  # else:
  #   threshold = np.sort(value_array)[int(0.1*n_data)]
  #   pred = np.zeros(n_data)
  #   pred[value_array < threshold] = 1
    
  true = np.zeros(n_data)
  true[:int(0.1*n_data)] = 1
  return roc_auc_score( true, - value_array )

