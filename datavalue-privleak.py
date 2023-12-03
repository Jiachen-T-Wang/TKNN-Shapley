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

# python datavalue-privleak.py --dataset cpu --value_type KNN-SV-JW --n_data 200 --n_val 200 --flip_ratio 0 --random_state 1 --K 5

parser = argparse.ArgumentParser('')

parser.add_argument('--dataset', type=str)
parser.add_argument('--value_type', type=str)
parser.add_argument('--model_type', type=str)
parser.add_argument('--n_data', type=int, default=500)
parser.add_argument('--n_val', type=int, default=2000)
parser.add_argument('--n_repeat', type=int, default=5)
parser.add_argument('--n_sample', type=int)
parser.add_argument('--random_state', type=int)
parser.add_argument('--flip_ratio', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--alpha', type=int, default=1)
parser.add_argument('--beta', type=int, default=1)
parser.add_argument('--noise', type=float, default=0)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--card', type=int, default=0)
parser.add_argument('--last_epoch', action='store_false')

parser.add_argument('--K', type=int, default=5)
parser.add_argument('--tau', type=float, default=0)
parser.add_argument('--sigma', type=float, default=0)
parser.add_argument('--q', type=float, default=1)
parser.add_argument('--delta', type=float, default=1e-5)
parser.add_argument('--eps', type=float, default=0)
parser.add_argument('--q_test', type=float, default=1)


args = parser.parse_args()

dataset = args.dataset
value_type = args.value_type
model_type = args.model_type
n_data = args.n_data
n_val = args.n_val
n_repeat = args.n_repeat
n_sample = args.n_sample
random_state = args.random_state
flip_ratio = float(args.flip_ratio) * 1.0
batch_size = args.batch_size
lr = args.lr
a, b = args.alpha, args.beta
card = args.card

K, tau = args.K, args.tau
sigma, q = args.sigma, args.q
delta = args.delta
eps = np.infty
q_test = args.q_test

big_dataset = config.big_dataset
OpenML_dataset = config.OpenML_dataset

save_dir = 'result/'

verbose = 0
if args.debug:
  verbose = 1

batch_size = 32
u_func = get_ufunc(dataset, model_type, batch_size, lr, verbose)

x_train, y_train, x_val, y_val = get_processed_data(dataset, n_data, n_val, flip_ratio)


if(random_state != -1): 
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)


knn_val_collection = ['KNN-SV', 'KNN-BZ', 'KNN-SV-RJ', 'KNN-SV-JW', 'KNN-BZ-private', 'KNN-SV-RJ-private', 'KNN-BZ-private-fixeps']


if value_type != 'Uniform' and value_type != 'inf' and value_type not in knn_val_collection:
  value_args = load_value_args(value_type, args)
  value_args['n_data'] = n_data
  if dataset in big_dataset:
    value_args['sv_baseline'] = 0.1
  else:
    value_args['sv_baseline'] = 0.5
else:
  value_args = {}
  value_args['n_data'] = n_data


data_lst = []

sv_collect = []



for i in range(1):
  
  print('Compute Data Value')
  
  v_args = copy.deepcopy(value_args)
  
  if value_type in ['Shapley_Perm', 'Banzhaf_GT', 'BetaShapley']:

    if dataset in big_dataset or dataset in OpenML_dataset :
      v_args['y_feature'] = value_args['y_feature'][:, i]
    else:
      v_args['y_feature'] = np.clip( value_args['y_feature'] + np.random.normal(scale=args.noise, size=n_sample) , a_min=0, a_max=1)

  elif value_type == 'LOO':

    if dataset in big_dataset or dataset in OpenML_dataset :
      v_args['y_feature'] = value_args['y_feature'][:, i]
      v_args['u_total'] = value_args['u_total'][i]
    else:
      v_args['y_feature'] = np.clip( value_args['y_feature']+np.random.normal(scale=args.noise, size=len(value_args['y_feature'])), a_min=0, a_max=1)
      v_args['u_total'] = np.clip( value_args['u_total']+np.random.normal(scale=args.noise), a_min=0, a_max=1)

  elif value_type == 'FixedCard_MC':

    def func(x):
      if type(x[i]) is list:
        if args.last_epoch:
          return x[i][-1]
        else:
          return get_converge(x[i])
      else:
        return x[i]

    v_args['func'] = func

  elif value_type in ['FixedCard_MSR', 'FixedCard_MSRPerm']:
    
    def func(x):
      if type(x[i]) is list:
        if args.last_epoch:
          return x[i][-1]
        else:
          return get_converge(x[i])
      else:
        return x[i]

    v_args['func'] = func

  elif value_type in ['FZ20']:

    if args.dataset in OpenML_dataset:
      v_args['y_feature'] = value_args['y_feature'][:, i]
    else:
      v_args['y_feature'] = value_args['y_feature'][:, i, -1]


  if value_type == 'KNN-SV-RJ':
    knn_shapley = knn_shapley_RJ
  elif value_type == 'KNN-SV-JW':
    knn_shapley = knn_shapley_JW
  else:
    exit(1)

  sv_lst = []

  for _ in range(1):
      x_train_aug = np.insert(x_train, 0, x_train[0], axis=0)
      y_train_aug = np.insert(y_train, 0, y_train[0])

  print(x_train_aug.shape)

  for K in range(1, 20):
    print(K)
    sv1 = knn_shapley(x_train, y_train, x_val, y_val, K=K)
    sv2 = knn_shapley(x_train_aug, y_train_aug, x_val, y_val, K=K)
    sv_lst.append([sv1, sv2])

  distance = np.array([np.linalg.norm(x - x_train[0]) for x in x_train])
  rank = np.argsort(distance)

  pickle.dump( [sv_lst, rank], open('/home/tw8948/private-knnsv/HOTEL-value-privleak_{}_{}_{}-diff1.data'.format(dataset, value_type, n_data), 'wb') )



  





