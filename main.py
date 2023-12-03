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
import pdb

from helper import *
from helper_knn import *
from utility_func import *
from prepare_data import *
from if_utils import *
import config
import argparse

parser = argparse.ArgumentParser('')

parser.add_argument('--dataset', type=str)
parser.add_argument('--value_type', type=str)
parser.add_argument('--n_data', type=int, default=2000)
parser.add_argument('--n_val', type=int, default=200)
parser.add_argument('--flip_ratio', type=float, default=0)
parser.add_argument('--task', type=str)
parser.add_argument('--dis_metric', type=str, default='cosine')
parser.add_argument('--K', type=int, default=5)
parser.add_argument('--tau', type=float, default=0)

# Hyperparameters for privacy-preserving setting (Gaussian mechanism)
parser.add_argument('--sigma', type=float, default=0)
parser.add_argument('--q', type=float, default=1)
parser.add_argument('--eps', type=float, default=-1)
parser.add_argument('--delta', type=float, default=1e-5)
parser.add_argument('--q_test', type=float, default=1)
parser.add_argument('--n_repeat', type=int, default=5)

args = parser.parse_args()

dataset = args.dataset
value_type = args.value_type
n_data = args.n_data
n_val = args.n_val
flip_ratio = float(args.flip_ratio) * 1.0
task = args.task
dis_metric = args.dis_metric
K, tau = args.K, args.tau

sigma, q = args.sigma, args.q
delta = args.delta
eps = args.eps
q_test = args.q_test
n_repeat = args.n_repeat


assert sigma==0 or eps==0


big_dataset = config.big_dataset
OpenML_dataset = config.OpenML_dataset

save_dir = 'result/'

if task=='mislabel_detect':
  x_train, y_train, x_val, y_val = get_processed_data(dataset, n_data, n_val, flip_ratio, noisy_data=False)
elif task=='noisy_detect':
  x_train, y_train, x_val, y_val = get_processed_data(dataset, n_data, n_val, flip_ratio, noisy_data=True)
else:
  exit(1)


data_lst = []


for i in range(n_repeat):
  
  if args.eps > 0 and i==0:
    sigma_use = -1
  elif args.eps < 0:
    sigma_use = sigma
  else:
    pass

  start = time.time()

  if value_type == 'KNN-SV-RJ':
    sv = knn_shapley_RJ(x_train, y_train, x_val, y_val, K=K, dis_metric=dis_metric )
  elif value_type == 'KNN-SV-RJ-private':
    sv, eps, sigma_use = private_knn_shapley_RJ(x_train, y_train, x_val, y_val, K=K, sigma=sigma_use, delta=delta, q_test=q_test, dis_metric=dis_metric, eps=args.eps)
  elif value_type == 'KNN-SV-RJ-private-withsub':
    sv, eps, sigma_use = private_knn_shapley_RJ_withsub(x_train, y_train, x_val, y_val, K=K, sigma=sigma_use, q=q, delta=delta, q_test=q_test, dis_metric=dis_metric, eps=args.eps)
  elif value_type == 'KNN-SV-JW':
    sv = knn_shapley_JW(x_train, y_train, x_val, y_val, K=K, dis_metric = dis_metric)
  elif value_type == 'TNN-SV':
    sv = tnn_shapley(x_train, y_train, x_val, y_val, tau=tau, K0=K, dis_metric=dis_metric)
  elif value_type == 'TNN-SV-private':
    sv, eps, sigma_use = private_tnn_shapley_JDP(x_train, y_train, x_val, y_val, tau=tau, K0=K, sigma=sigma_use, q=q, delta=delta, q_test=q_test, dis_metric=dis_metric, eps=args.eps)
  else:
    exit(1)

  print('Data Value Computed; Value Name: {}; Runtime: {} s'.format( value_type, np.round(time.time()-start, 3) ))

  if task in ['mislabel_detect', 'noisy_detect']:
    auc = kmeans_aucroc(sv)
    data_lst.append( [auc] )


if task in ['mislabel_detect', 'noisy_detect']:

  print('Task: {}'.format(task))
  
  data_lst = np.array(data_lst)

  auc, std_auc = np.round( np.mean(data_lst[:, 0]), 3), np.round( np.std(data_lst[:, 0]), 3)

  if value_type in ['KNN-SV-RJ', 'KNN-SV-JW', 'TNN-SV']:
    eps, delta = np.inf, 0

  print('*** {} AUROC: {} ({}), eps={}, delta={}***'.format(value_type, auc, std_auc, eps, delta))