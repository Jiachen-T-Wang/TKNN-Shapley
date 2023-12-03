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


# python applications_knn.py --task mislabel_detect --dataset creditcard --value_type KNN-SV-JW --n_data 2000 --n_val 2000 --flip_ratio 0.1 --K 5

import argparse

parser = argparse.ArgumentParser('')

parser.add_argument('--dataset', type=str)
parser.add_argument('--value_type', type=str)
parser.add_argument('--n_data', type=int, default=500)
parser.add_argument('--n_val', type=int, default=2000)
parser.add_argument('--flip_ratio', type=float, default=0)
parser.add_argument('--task', type=str)
parser.add_argument('--dis_metric', type=str)
parser.add_argument('--K', type=int, default=5)
parser.add_argument('--tau', type=float, default=0)

parser.add_argument('--sigma', type=float, default=0)
parser.add_argument('--q', type=float, default=1)
parser.add_argument('--delta', type=float, default=1e-5)
parser.add_argument('--eps', type=float, default=0)
parser.add_argument('--q_test', type=float, default=1)
parser.add_argument('--val_corrupt', type=float, default=0)
parser.add_argument('--normalizerow', action='store_true')

parser.add_argument('--dim', type=int)

# No use for this project
parser.add_argument('--model_type', type=str, default='')
parser.add_argument('--n_repeat', type=int, default=5)
parser.add_argument('--n_sample', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--alpha', type=int, default=1)
parser.add_argument('--beta', type=int, default=1)
parser.add_argument('--noise', type=float, default=0)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--card', type=int, default=0)
parser.add_argument('--last_epoch', action='store_false')
parser.add_argument('--random_state', type=int, default=1)


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
task = args.task
card = args.card
K, tau = args.K, args.tau
sigma, q = args.sigma, args.q
delta = args.delta
eps = np.infty
q_test = args.q_test
dis_metric = args.dis_metric
big_dataset = config.big_dataset
OpenML_dataset = config.OpenML_dataset

save_dir = 'runtime_result/'

verbose = 0
if args.debug:
  verbose = 1


if task=='mislabel_detect':
  x_train, y_train, x_val, y_val = get_processed_data(dataset, n_data, n_val, flip_ratio, noisy_data=False, dim=args.dim)
elif task=='noisy_detect':
  x_train, y_train, x_val, y_val = get_processed_data_noisy(dataset, n_data, n_val, flip_ratio)


if args.normalizerow and dataset in OpenML_dataset:
  x_train, y_train, x_val, y_val = get_processed_data_clip(dataset, n_data, n_val, flip_ratio)


if args.val_corrupt > 0:
  n_corrupt = int(n_val * args.val_corrupt)
  x_val[:n_corrupt] += np.random.normal(loc=10.0, scale=0.0, size=(n_corrupt, x_val.shape[1]))


if value_type[:3] == 'TNN' and args.tau==0:
  threshold = get_tuned_tau(x_train, y_train, x_val, y_val, dis_metric = dis_metric)
  print('Tuned Tau:', threshold)
  tau = threshold


if value_type[:3] == 'KNN' and args.K==0:
  threshold = get_tuned_K(x_train, y_train, x_val, y_val, dis_metric=dis_metric)
  print('Tuned K:', threshold)
  K = threshold


if(random_state != -1): 
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)


knn_val_collection = ['KNN-SV', 'KNN-SV-RJ', 'KNN-SV-JW', 'TNN-BZ', 'TNN-BZ-private', 'TNN-SV', 'TNN-SV-private', 'KNN-SV-RJ-private', 'KNN-BZ-private-fixeps']


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


time_lst = []


for i in range(5):

  start = time.time()

  if value_type == 'inf':
    sv = compute_influence_score(dataset, model_type, x_train, y_train, x_val, y_val, batch_size, lr, verbose=0)
  elif value_type == 'KNN-SV-RJ':
    sv = knn_shapley_RJ(x_train, y_train, x_val, y_val, K=K, dis_metric=dis_metric )
  elif value_type == 'KNN-SV-RJ-private':
    sv, eps = private_knn_shapley_RJ(x_train, y_train, x_val, y_val, K=K, sigma=sigma, q=q, delta=delta, q_test=q_test, dis_metric=dis_metric)
  elif value_type == 'KNN-SV-JW':
    sv = knn_shapley_JW(x_train, y_train, x_val, y_val, K=K, dis_metric = dis_metric)
  elif value_type == 'TNN-BZ':
    sv = knn_banzhaf(x_train, y_train, x_val, y_val, tau=tau, K0=K, dis_metric=dis_metric)
  elif value_type == 'TNN-SV':
    sv = tnn_shapley(x_train, y_train, x_val, y_val, tau=tau, K0=K, dis_metric=dis_metric)
  elif value_type == 'TNN-BZ-private':
    sv, eps = private_knn_banzhaf(x_train, y_train, x_val, y_val, tau=tau, K0=K, sigma=sigma, q=q, delta=delta, q_test=q_test, dis_metric = dis_metric)
  elif value_type == 'TNN-SV-private':
    sv, eps = private_tnn_shapley(x_train, y_train, x_val, y_val, tau=tau, K0=K, sigma=sigma, q=q, delta=delta, q_test=q_test, debug=args.debug, dis_metric = dis_metric)
  else:
    exit(1)

  print('Data Value Computed; Value Name: {}; Runtime: {} s'.format( value_type, np.round(time.time()-start, 3) ))

  time_lst.append( time.time()-start )


file_name = 'RUNTIME_{}_{}_{}_{}_DIM{}.result'.format(args.dataset, args.value_type, args.n_data, args.n_val, args.dim)

pickle.dump( time_lst, open(save_dir + file_name, 'wb') )
