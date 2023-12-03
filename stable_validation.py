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
from utility_func import *
from prepare_data import *
from if_utils import *
import config


# python applications.py --task data_removal --dataset MNIST --value_type inf --model_type SmallCNN --n_data 2000 --n_val 2000 --n_repeat 5 --n_sample 10000 --batch_size 128 --lr 1e-3 --flip_ratio 0.1 --card 15 --random_state 1

# python applications.py --task data_removal --dataset cpu --value_type inf --model_type MLP --n_data 200 --n_val 200 --n_repeat 5 --n_sample 10000 --batch_size 128 --lr 1e-2 --flip_ratio 0.1 --card 5 --random_state 1

# python applications.py --task collect_sv --dataset Dog_vs_CatFeature --value_type FixedCard_MSRPerm --model_type MLP --n_data 2000 --n_val 2000 --n_repeat 5 --n_sample 10000 --batch_size 128 --lr 1e-3 --flip_ratio 0.1 --card 15 --random_state 1

# python applications.py --task mislabel_detect --dataset Dog_vs_CatFeature --value_type FZ20 --model_type MLP --n_data 2000 --n_val 2000 --n_repeat 5 --n_sample 10000 --batch_size 128 --lr 1e-3 --flip_ratio 0.1 --card 10 --random_state 1

# python applications_knn.py --task mislabel_detect --dataset cpu --value_type KNN-SV --model_type MLP --n_data 200 --n_val 200 --n_repeat 5 --n_sample 10000 --batch_size 128 --lr 1e-2 --flip_ratio 0.1 --card 5 --random_state 1

# python stable_validation.py --task collect_sv --dataset creditcard --value_type KNN-SV-JW --n_data 1000 --n_val 200 --flip_ratio 0.1 --random_state 1 --K 5

# python stable_validation.py --task mislabel_detect --dataset creditcard --value_type KNN-SV-JW --n_data 1000 --n_val 10 --flip_ratio 0.1 --random_state 1 --K 5 >> knn_result/Creditcard_KNNSVJW_Val_N1000.txt


import argparse

parser = argparse.ArgumentParser('')

parser.add_argument('--dataset', type=str)
parser.add_argument('--value_type', type=str)
parser.add_argument('--model_type', type=str, default='')
parser.add_argument('--n_data', type=int, default=500)
parser.add_argument('--n_val', type=int, default=2000)
parser.add_argument('--n_repeat', type=int, default=5)
parser.add_argument('--n_sample', type=int, default=0)
parser.add_argument('--random_state', type=int)
parser.add_argument('--flip_ratio', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--alpha', type=int, default=1)
parser.add_argument('--beta', type=int, default=1)
parser.add_argument('--noise', type=float, default=0)
parser.add_argument('--task', type=str)
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

parser.add_argument('--val_corrupt', type=float, default=0)
parser.add_argument('--normalizerow', action='store_true')



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

big_dataset = config.big_dataset
OpenML_dataset = config.OpenML_dataset

save_dir = 'result/'

verbose = 0
if args.debug:
  verbose = 1

batch_size = 32

if task != 'mislabel_detect':
  u_func = get_ufunc(dataset, model_type, batch_size, lr, verbose)


# Sample 5 times of the size of the intended validation data
if task in ['mislabel_detect', 'collect_sv']:
  x_train, y_train, x_val, y_val = get_processed_data(dataset, n_data, n_val*5, flip_ratio)
elif task=='noisy_detect':
  x_train, y_train, x_val, y_val = get_processed_data_noisy(dataset, n_data, n_val*5, flip_ratio)


if args.normalizerow and dataset in OpenML_dataset:
  x_train, y_train, x_val, y_val = get_processed_data_clip(dataset, n_data, n_val*5, flip_ratio)


if args.val_corrupt > 0:
  n_corrupt = int(n_val * args.val_corrupt)
  x_val[:n_corrupt] += np.random.normal(loc=10.0, scale=0.0, size=(n_corrupt, x_val.shape[1]))


if value_type[:3] == 'TNN' and args.tau==0:
  threshold = get_tuned_tau(x_train, y_train, x_val, y_val)
  print('Tuned Tau:', threshold)
  tau = threshold


if value_type[:3] == 'KNN' and args.K==0:
  threshold = get_tuned_K(x_train, y_train, x_val, y_val)
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


data_lst = []

sv_collect = []



x_val_all, y_val_all = x_val, y_val


for i in range(5):
  
  print('iter i={}'.format(i))
  
  x_val, y_val = x_val_all[i*n_val:(i+1)*n_val], y_val_all[i*n_val:(i+1)*n_val]

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


  if value_type == 'inf':
    sv = compute_influence_score(dataset, model_type, x_train, y_train, x_val, y_val, batch_size, lr, verbose=0)
  elif value_type == 'KNN-SV-RJ':
    sv = knn_shapley_RJ(x_train, y_train, x_val, y_val, K=K)
  elif value_type == 'KNN-SV-RJ-private':
    sv, eps = private_knn_shapley_RJ(x_train, y_train, x_val, y_val, K=K, sigma=sigma, q=q, delta=delta, q_test=q_test)
  elif value_type == 'KNN-SV-JW':
    sv = knn_shapley_JW(x_train, y_train, x_val, y_val, K=K)
  elif value_type == 'TNN-BZ':
    sv = knn_banzhaf(x_train, y_train, x_val, y_val, tau=tau, K0=K)
  elif value_type == 'TNN-SV':
    sv = tnn_shapley(x_train, y_train, x_val, y_val, tau=tau, K0=K)
  elif value_type == 'TNN-BZ-private':
    sv, eps = private_knn_banzhaf(x_train, y_train, x_val, y_val, tau=tau, K0=K, sigma=sigma, q=q, delta=delta, q_test=q_test)
  elif value_type == 'TNN-SV-private':
    sv, eps = private_tnn_shapley(x_train, y_train, x_val, y_val, tau=tau, K0=K, sigma=sigma, q=q, delta=delta, q_test=q_test)
  else:
    sv = compute_value(value_type, v_args)

  if task=='weighted_acc':

    sv = normalize(sv) if value_type!='Uniform' else sv

    if dataset in big_dataset or dataset in OpenML_dataset:
      acc_lst = []
      for j in range(5):
        acc_lst.append( u_func(x_train, y_train, x_val, y_val, sv) )
      acc = np.mean(acc_lst)
    else:
      acc = u_func(x_train, y_train, x_val, y_val, sv)
    print('round {}, acc={}'.format(i, acc))
    data_lst.append( acc )
    

  elif task in ['mislabel_detect', 'noisy_detect']:
    # acc1, acc2 = kmeans_f1score(sv, cluster=False), kmeans_f1score(sv, cluster=True)

    acc1, acc2 = kmeans_aucroc(sv), kmeans_aucroc(sv)
    data_lst.append( [acc1, acc2] )

    
  elif task=='data_removal':

    rank = np.argsort(sv)
    acc_lst = []
    
    for k in np.linspace(0, int(args.n_data/2), num=11).astype(int):

      temp_lst = []
      for j in range(5):
        temp_lst.append( u_func(x_train[rank[k:]], y_train[rank[k:]], x_val, y_val) )
      acc = np.mean(temp_lst)
      print(acc)
      acc_lst.append(acc)
      
    print(acc_lst)
      
    data_lst.append(acc_lst)


  elif task=='data_add':

    rank = np.argsort(sv)[::-1]
    acc_lst = []

    # pdb.set_trace()
    
    for k in np.linspace(0, int(args.n_data/2), num=11).astype(int)[1:]:

      temp_lst = []
      for j in range(5):
        temp_lst.append( u_func(x_train[rank[:k]], y_train[rank[:k]], x_val, y_val) )
      acc = np.mean(temp_lst)
      acc_lst.append(acc)
      
    print(acc_lst)
      
    data_lst.append(acc_lst)

  elif task=='collect_sv':
    sv_collect.append(sv)



if task in ['mislabel_detect', 'noisy_detect']:
  
  data_lst = np.array(data_lst)

  acc_nocluster, std_nocluster = np.round( np.mean(data_lst[:, 0]), 3), np.round( np.std(data_lst[:, 0]), 3)
  acc_cluster, std_cluster = np.round( np.mean(data_lst[:, 1]), 3), np.round( np.std(data_lst[:, 1]), 3)

  if value_type == 'BetaShapley':
    print('*** {}_{}_{} {} ({}) {} ({}) ***'.format(value_type, a, b, acc_nocluster, std_nocluster, acc_cluster, std_cluster ))
  elif value_type in ['FixedCard_MC', 'FixedCard_MSR', 'FixedCard_MSRPerm']:
    print('*** {} card={} {} ({}) {} ({}) ***'.format(value_type, args.card, acc_nocluster, std_nocluster, acc_cluster, std_cluster ))
  else:
    print('*** {} {} ({}) {} ({}), eps={}, delta={} ***'.format(value_type, acc_nocluster, std_nocluster, acc_cluster, std_cluster, np.mean(eps), delta))
    
elif task == 'data_removal':
  
  file_name = 'DATAREMOVAL_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.result'.format(args.dataset, args.value_type, args.model_type, args.n_data, args.n_val, 
                                                                           args.n_repeat, args.n_sample, args.flip_ratio, args.alpha, args.beta, args.card )
  
  pickle.dump( data_lst, open(save_dir + file_name, 'wb') )

elif task == 'data_add':
  
  file_name = 'DATAADD_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.result'.format(args.dataset, args.value_type, args.model_type, args.n_data, args.n_val, 
                                                                           args.n_repeat, args.n_sample, args.flip_ratio, args.alpha, args.beta, args.card )
  
  pickle.dump( data_lst, open(save_dir + file_name, 'wb') )


elif task == 'collect_sv':

  file_name = 'SV_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.result'.format(args.dataset, args.value_type, args.model_type, args.n_data, args.n_val, 
                                                                           args.n_repeat, args.n_sample, args.flip_ratio, args.alpha, args.beta, args.card )
  
  pickle.dump( sv_collect, open(save_dir + file_name, 'wb') )



