# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.autograd import Variable
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, TensorDataset

# import torchvision.datasets as datasets
# import torchvision
# import torchvision.transforms as transforms

# general
import pandas as pd 
import numpy as np 
import copy
import pickle
import sys, os
import time
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

big_dataset = config.big_dataset
OpenML_dataset = config.OpenML_dataset

save_dir = 'result/'

verbose = 0
if args.debug:
  verbose = 1


if task=='mislabel_detect':
  x_train, y_train, x_val, y_val = get_processed_data(dataset, n_data, n_val, flip_ratio, noisy_data=False)
elif task=='noisy_detect':
  x_train, y_train, x_val, y_val = get_processed_data(dataset, n_data, n_val, flip_ratio, noisy_data=True)

knn_val_collection = ['KNN-SV', 'KNN-SV-RJ', 'KNN-SV-JW', 'TNN-BZ', 'TNN-BZ-private', 'TNN-SV', 'TNN-SV-private', 'KNN-SV-RJ-private', 'KNN-SV-RJ-private-withsub', 'KNN-BZ-private-fixeps', 'TNN-SV-private-JDP']


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

sigma_use = -1

for i in range(n_repeat):

  if(random_state != -1): 
      np.random.seed(random_state+i)
      random.seed(random_state+i)

  v_args = copy.deepcopy(value_args)

  if value_type in ['Shapley_Perm', 'Banzhaf_GT', 'BetaShapley']:

    if dataset in big_dataset or dataset in OpenML_dataset :
      v_args['y_feature'] = value_args['y_feature']
    else:
      v_args['y_feature'] = np.clip( value_args['y_feature'] + np.random.normal(scale=args.noise, size=n_sample) , a_min=0, a_max=1)

  elif value_type == 'LOO':

    if dataset in big_dataset or dataset in OpenML_dataset :
      v_args['y_feature'] = value_args['y_feature']
      v_args['u_total'] = value_args['u_total']
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


  start = time.time()

  if args.eps > 0 and i==0:
    sigma_use = -1
  elif args.eps < 0:
    sigma_use = sigma
  else:
    pass

  if value_type == 'inf':
    sv = compute_influence_score(dataset, model_type, x_train, y_train, x_val, y_val, batch_size, lr, verbose=0)
  elif value_type == 'KNN-SV-RJ':
    sv = knn_shapley_RJ(x_train, y_train, x_val, y_val, K=K, dis_metric=dis_metric )
  elif value_type == 'KNN-SV-RJ-private':
    sv, eps, sigma_use = private_knn_shapley_RJ(x_train, y_train, x_val, y_val, K=K, sigma=sigma_use, delta=delta, q_test=q_test, dis_metric=dis_metric, eps=args.eps)
  elif value_type == 'KNN-SV-RJ-private-withsub':
    sv, eps, sigma_use = private_knn_shapley_RJ_withsub(x_train, y_train, x_val, y_val, K=K, sigma=sigma_use, q=q, delta=delta, q_test=q_test, dis_metric=dis_metric, eps=args.eps)
  elif value_type == 'KNN-SV-JW':
    sv = knn_shapley_JW(x_train, y_train, x_val, y_val, K=K, dis_metric = dis_metric)
  elif value_type == 'TNN-BZ':
    sv = knn_banzhaf(x_train, y_train, x_val, y_val, tau=tau, K0=K, dis_metric=dis_metric)
  elif value_type == 'TNN-SV':
    sv = tnn_shapley(x_train, y_train, x_val, y_val, tau=tau, K0=K, dis_metric=dis_metric)
  elif value_type == 'TNN-BZ-private':
    sv, eps, sigma_use = private_knn_banzhaf(x_train, y_train, x_val, y_val, tau=tau, K0=K, sigma=sigma_use, q=q, delta=delta, q_test=q_test, dis_metric = dis_metric)
  elif value_type == 'TNN-SV-private':
    sv, eps, sigma_use = private_tnn_shapley(x_train, y_train, x_val, y_val, tau=tau, K0=K, sigma=sigma_use, q=q, delta=delta, q_test=q_test, debug=args.debug, dis_metric = dis_metric)
  elif value_type == 'TNN-SV-private-JDP':
    sv, eps, sigma_use = private_tnn_shapley_JDP(x_train, y_train, x_val, y_val, tau=tau, K0=K, sigma=sigma_use, q=q, delta=delta, q_test=q_test, dis_metric=dis_metric, eps=args.eps)
  else:
    sv = compute_value(value_type, v_args)

  print('Data Value Computed; Value Name: {}; Runtime: {} s'.format( value_type, np.round(time.time()-start, 3) ))


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
    acc1, acc2, auc = kmeans_f1score(sv, cluster=False), kmeans_f1score(sv, cluster=True), kmeans_aucroc(sv)
    data_lst.append( [acc1, acc2, auc] )

    
  elif task=='data_removal':

    rank = np.argsort(sv)
    acc_lst = []
    
    # Only remove at most 20% of the data points. 
    for k in np.linspace(0, int(args.n_data * 0.2), num=11).astype(int):

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
    
    for k in np.linspace(0, int(args.n_data), num=21).astype(int)[1:]:

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

  print('Task: {}'.format(task))
  
  data_lst = np.array(data_lst)

  f1_rank, f1_rank_std = np.round( np.mean(data_lst[:, 0]), 3), np.round( np.std(data_lst[:, 0]), 3)
  f1_cluster, f1_cluster_std = np.round( np.mean(data_lst[:, 1]), 3), np.round( np.std(data_lst[:, 1]), 3)
  auc, std_auc = np.round( np.mean(data_lst[:, 2]), 3), np.round( np.std(data_lst[:, 2]), 3)

  if value_type == 'BetaShapley':
    print('*** {}_{}_{} {} ({}) {} ({}) ***'.format(value_type, a, b, f1_rank, f1_rank_std, f1_cluster, f1_cluster_std ))
  elif value_type in ['FixedCard_MC', 'FixedCard_MSR', 'FixedCard_MSRPerm']:
    print('*** {} card={} {} ({}) {} ({}) ***'.format(value_type, args.card, f1_rank, f1_rank_std, f1_cluster, f1_cluster_std ))
  else:
    print('*** {} F1-Rank: {} ({}), F1-Cluster: {} ({}), AUROC: {} ({}), eps={}, delta={}, K={}, tau={} ***'.format(
      value_type, f1_rank, f1_rank_std, f1_cluster, f1_cluster_std, auc, std_auc, eps, delta, K, tau))
    
elif task == 'data_removal':
  
  file_name = 'DATAREMOVAL_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.result'.format(args.dataset, args.value_type, args.model_type, args.n_data, args.n_val, 
                                                                        args.n_repeat, args.n_sample, args.flip_ratio, args.eps, args.q)
  
  pickle.dump( data_lst, open('dataremoval/' + file_name, 'wb') )

elif task == 'data_add':
  
  file_name = 'DATAADD_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.result'.format(args.dataset, args.value_type, args.model_type, args.n_data, args.n_val, 
                                                                        args.n_repeat, args.n_sample, args.flip_ratio, args.eps, args.q)
  
  pickle.dump( data_lst, open('dataadd/' + file_name, 'wb') )

elif task == 'collect_sv':

  file_name = 'SV_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.result'.format(args.dataset, args.value_type, args.model_type, args.n_data, args.n_val, 
                                                                           args.n_repeat, args.n_sample, args.flip_ratio, args.alpha, args.beta, args.card )
  
  pickle.dump( sv_collect, open(save_dir + file_name, 'wb') )

