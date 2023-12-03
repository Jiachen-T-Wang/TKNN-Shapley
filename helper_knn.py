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



def rank_neighbor(x_test, x_train, dis_metric='cosine'):
  if dis_metric == 'cosine':
    distance = -np.dot(x_train, x_test) / np.linalg.norm(x_train, axis=1)
  else:
    distance = np.array([np.linalg.norm(x - x_test) for x in x_train])
  return np.argsort(distance)


# x_test, y_test are single data point
def knn_shapley_RJ_single(x_train_few, y_train_few, x_test, y_test, K, dis_metric='cosine'):
  N = len(y_train_few)
  sv = np.zeros(N)
  rank = rank_neighbor(x_test, x_train_few, dis_metric = dis_metric)
  sv[int(rank[-1])] += int(y_test==y_train_few[int(rank[-1])]) / N

  for j in range(2, N+1):
    i = N+1-j
    sv[int(rank[-j])] = sv[int(rank[-(j-1)])] + ( (int(y_test==y_train_few[int(rank[-j])]) - int(y_test==y_train_few[int(rank[-(j-1)])])) / K ) * min(K, i) / i

  return sv


# Original KNN-Shapley proposed in http://www.vldb.org/pvldb/vol12/p1610-jia.pdf
def knn_shapley_RJ(x_train_few, y_train_few, x_val_few, y_val_few, K, dis_metric='cosine'):
  
  N = len(y_train_few)
  sv = np.zeros(N)

  n_test = len(y_val_few)
  for i in range(n_test):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv += knn_shapley_RJ_single(x_train_few, y_train_few, x_test, y_test, K, dis_metric='cosine')

  return sv


# x_test, y_test are single data point
def knn_shapley_JW_single(x_train_few, y_train_few, x_test, y_test, K, dis_metric='cosine'):
  N = len(y_train_few)
  sv = np.zeros(N)
  rank = rank_neighbor(x_test, x_train_few, dis_metric=dis_metric).astype(int)
  C = max(y_train_few)+1

  c_A = np.sum( y_test==y_train_few[rank[:N-1]] )

  const = np.sum([ 1/j for j in range(1, min(K, N)+1) ])

  sv[rank[-1]] = (int(y_test==y_train_few[rank[-1]]) - c_A/(N-1)) / N * ( np.sum([ 1/(j+1) for j in range(1, min(K, N)) ]) ) + (int(y_test==y_train_few[rank[-1]]) - 1/C) / N

  for j in range(2, N+1):
    i = N+1-j
    coef = (int(y_test==y_train_few[int(rank[-j])]) - int(y_test==y_train_few[int(rank[-(j-1)])])) / (N-1)

    sum_K3 = K

    sv[int(rank[-j])] = sv[int(rank[-(j-1)])] + coef * ( const + int( N >= K ) / K * ( min(i, K)*(N-1)/i - sum_K3 ) )

  return sv


# Soft-label KNN-Shapley proposed in https://arxiv.org/abs/2304.04258 
def knn_shapley_JW(x_train_few, y_train_few, x_val_few, y_val_few, K, dis_metric='cosine'):
  
  N = len(y_train_few)
  sv = np.zeros(N)

  n_test = len(y_val_few)
  for i in range(n_test):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv += knn_shapley_JW_single(x_train_few, y_train_few, x_test, y_test, K, dis_metric = dis_metric)

  return sv


def get_knn_acc(x_train, y_train, x_val, y_val, K, dis_metric='cosine'):
  n_val = len(y_val)
  C = max(y_train)+1

  acc = 0

  for i in range(n_val):
    x_test, y_test = x_val[i], y_val[i]
    if dis_metric == 'cosine':
      distance = -np.dot(x_train, x_test)
    else:
      distance = np.array([np.linalg.norm(x - x_test) for x in x_train])
    rank = np.argsort(distance)
    acc_single = 0
    for j in range(K):
      acc_single += int(y_test==y_train[ rank[j] ])
    acc += (acc_single/K)

  return acc / n_val



def get_tuned_K(x_train, y_train, x_val, y_val, dis_metric='cosine'):

  acc_max = 0
  best_K = 0

  for K in range(1, 8):
    acc = get_knn_acc(x_train, y_train, x_val, y_val, K, dis_metric = dis_metric)
    print('K={}, acc={}'.format(K, acc))
    if acc > acc_max:
      acc_max = acc
      best_K = K

  return best_K


def get_tnn_acc(x_train, y_train, x_val, y_val, tau, dis_metric='cosine'):
  n_val = len(y_val)
  C = max(y_train)+1
  acc = 0
  for i in range(n_val):
    x_test, y_test = x_val[i], y_val[i]
    #ix_test = x_test.reshape((-1,1))
    if dis_metric == 'cosine':
      distance = - np.dot(x_train, x_test) / np.linalg.norm(x_train, axis=1)


      #print('distance[:10]', distance[:10])
    else:
      distance = np.array([np.linalg.norm(x - x_test) for x in x_train])
    Itau = (distance<tau).nonzero()[0]
    acc_single = 0
    #print(f'tune tau size of Tau is {len(Itau)}')
    if len(Itau) > 0:
      for j in Itau:
        acc_single += int(y_test==y_train[j])
      acc_single = acc_single / len(Itau)
    else:
      acc_single = 1/C
    acc += acc_single

  return acc / n_val

def get_tuned_tau(x_train, y_train, x_val, y_val, dis_metric='cosine'):

  print('dis_metric', dis_metric)
  acc_max = 0
  best_tau = 0
  # because we use the negative cosine value as the distance metric
  tau_list =[-0.04*x for x in range(25)]+[0.04*x for x in range(10)]
  for tau in tau_list:
    acc = get_tnn_acc(x_train, y_train, x_val, y_val, tau, dis_metric=dis_metric)
    print('tau={}, acc={}'.format(tau, acc))
    if acc > acc_max:
      acc_max = acc
      best_tau = tau

  if best_tau == 1:
    for tau in (np.arange(1, 10) / 10):
      acc = get_tnn_acc(x_train, y_train, x_val, y_val, tau, dis_metric=dis_metric)
      print('tau={}, acc={}'.format(tau, acc))
      if acc > acc_max:
        acc_max = acc
        best_tau = tau

  return best_tau


# x_test, y_test are single data point
def tnn_shapley_single(x_train_few, y_train_few, x_test, y_test, tau=0, K0=10, dis_metric='cosine'):

  N = len(y_train_few)
  sv = np.zeros(N)

  C = max(y_train_few)+1
  if dis_metric == 'cosine':
    distance = -np.dot(x_train_few, x_test) / np.linalg.norm( x_train_few, axis=1 )
  else:
    distance = np.array([np.linalg.norm(x - x_test) for x in x_train_few])
  Itau = (distance < tau).nonzero()[0]

  Ct = len(Itau)
  Ca = np.sum( y_train_few[Itau] == y_test )

  reusable_sum = 0
  stable_ratio = 1
  for j in range(N):
    stable_ratio *= (N-j-Ct) / (N-j)
    reusable_sum += (1/(j+1)) * (1 - stable_ratio)
    # reusable_sum += (1/(j+1)) * (1 - comb(N-1-j, Ct) / comb(N, Ct))

  for i in Itau:
    sv[i] = ( int(y_test==y_train_few[i]) - 1/C ) / Ct
    if Ct >= 2:
      ca = Ca - int(y_test==y_train_few[i])
      sv[i] += ( int(y_test==y_train_few[i])/Ct - ca/(Ct*(Ct-1)) ) * ( reusable_sum - 1 )

  return sv

def tnn_shapley(x_train_few, y_train_few, x_val_few, y_val_few, tau=0, K0=10, dis_metric='cosine'):
  
  N = len(y_train_few)
  sv = np.zeros(N)
  n_test = len(y_val_few)
  print('tau in tnn shapley', tau)
  for i in tqdm(range(n_test)):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv += tnn_shapley_single(x_train_few, y_train_few, x_test, y_test, tau, K0, dis_metric=dis_metric)

  return sv



def private_tnn_shapley_single_JDP(x_train_few, y_train_few, x_test, y_test, Nsubsethat, tau=0, K0=10, sigma=0, q=1, dis_metric='cosine'):

  N = len(y_train_few)
  sv = np.zeros(N)
  C = max(y_train_few)+1

  # Poisson Subsampling
  sub_ind_bool = (np.random.choice([0, 1], size=N, p=[1-q, q])).astype(bool)
  sub_ind = np.where(sub_ind_bool)[0]
  if dis_metric == 'cosine':
    distance = -np.dot(x_train_few, x_test) / np.linalg.norm( x_train_few, axis=1 )
  else:
    distance = np.array([np.linalg.norm(x - x_test) for x in x_train_few])

  Itau_all = (distance <= tau).nonzero()[0]

  # Itau_subset: index in terms of subset
  distance_subset = distance[sub_ind]
  Itau_subset = (distance_subset <= tau).nonzero()[0]

  Ct = len(Itau_subset) + np.random.normal(scale=sigma)
  Ca = np.sum( y_train_few[sub_ind[Itau_subset]] == y_test ) + np.random.normal(scale=sigma)

  Ct, Ca = np.round(Ct), np.round(Ca)
  Ct, Ca = max(Ct, 0), max(Ca, 0)

  # N_subset = len(sub_ind)
  N_subset = Nsubsethat

  reusable_sum_i_in_sub = 0
  stable_ratio = 1
  for j in range(N_subset):
    stable_ratio *= (N_subset-j-max(1, Ct)) / (N_subset-j)
    reusable_sum_i_in_sub += (1/(j+1)) * (1 - stable_ratio)

  reusable_sum_i_notin_sub = 0
  stable_ratio = 1
  for j in range(N_subset+1):
    stable_ratio *= (N_subset+1-j-(Ct+1)) / (N_subset+1-j)
    reusable_sum_i_notin_sub += (1/(j+1)) * (1 - stable_ratio)

  for i in Itau_all:

      if i in sub_ind:
        reusable_sum = reusable_sum_i_in_sub
        Ct_i = max(1, Ct)
        Ca_i = Ca
        if y_test==y_train_few[i]:
          Ca_i = max(1, Ca_i)
      else:
        reusable_sum = reusable_sum_i_notin_sub
        Ct_i = Ct + 1
        Ca_i = Ca + int(y_test==y_train_few[i])

      sv[i] = ( int(y_test==y_train_few[i]) - 1/C ) / Ct_i
      if Ct_i >= 2:
        ca = Ca_i - int(y_test==y_train_few[i])
        sv[i] += ( int(y_test==y_train_few[i])/Ct_i - ca/(Ct_i*(Ct_i-1)) ) * ( reusable_sum - 1 )

  return sv


def private_tnn_shapley_JDP(x_train_few, y_train_few, x_val_few, y_val_few, tau=0, K0=10, sigma=0, q=1, delta=1e-5, q_test=0.1, dis_metric='cosine', rdp=False, eps=-1):
  
  N = len(y_train_few)
  sv = np.zeros(N)

  n_test = len(y_val_few)
  n_test_sub = int(n_test*q_test)
  test_ind = np.random.choice(range(n_test), size=n_test_sub, replace=False)
  x_val_few, y_val_few = x_val_few[test_ind], y_val_few[test_ind]

  n_compose = n_test_sub + 1

  if eps>0 and sigma<0:
    sigma = find_noise_multiplier(sampling_probability=q, num_steps=n_compose, target_epsilon=eps, target_delta=delta, eps_error=1e-2, mu_max=5000)
    print('Noise magnitude sigma={}'.format(sigma))
  elif eps<0:
    # First run RDP and get a rough estimate of eps
    # n_compose+1 since we need to count for the noisy N_subset
    mech = PrivateKNN_mech(q, sigma, n_compose)
    eps = mech.get_approxDP(delta=delta)

    # If eps estimate is too large or too small, use RDP
    if rdp or eps>30 or eps<0.01:
      print('Use RDP')
    else:
      print('Use PRV')
      prv = PoissonSubsampledGaussianMechanism(sampling_probability=q, noise_multiplier=sigma)
      acct = PRVAccountant(prvs=prv, max_self_compositions=n_compose, eps_error=1e-3, delta_error=1e-10)
      low, est, upp = acct.compute_epsilon(delta=delta, num_self_compositions=[n_compose])
      eps = upp
  else:
    pass
  
  sub_ind_bool = (np.random.choice([0, 1], size=N, p=[1-q, q])).astype(bool)
  sub_ind = np.where(sub_ind_bool)[0]
  N_subset = len(sub_ind)
  print('Noise magnitude sigma={}'.format(sigma))
  N_subset = np.round( N_subset + np.random.normal(scale=sigma) )
  N_subset = int( max(N_subset, 0) )

  for i in tqdm(range(n_test_sub)):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv_individual = private_tnn_shapley_single_JDP(x_train_few, y_train_few, x_test, y_test, N_subset, tau, K0, sigma*np.sqrt(2), q, dis_metric=dis_metric)
    sv += sv_individual

  return sv, eps, sigma


# x_test, y_test are single data point
def private_knn_shapley_RJ_single(x_train_few, y_train_few, x_test, y_test, K, sigma, dis_metric='cosine'):
  N = len(y_train_few)
  sv = np.zeros(N)
  rank = rank_neighbor(x_test, x_train_few, dis_metric=dis_metric)
  sv[int(rank[-1])] += int(y_test==y_train_few[int(rank[-1])]) / N + np.random.normal(scale=sigma)

  for j in range(2, N+1):
    i = N+1-j
    sv[int(rank[-j])] = sv[int(rank[-(j-1)])] + ( (int(y_test==y_train_few[int(rank[-j])]) - int(y_test==y_train_few[int(rank[-(j-1)])])) / K ) * min(K, i) / i + np.random.normal(scale=sigma)

  return sv

def private_knn_shapley_RJ(x_train_few, y_train_few, x_val_few, y_val_few, K, sigma=0, q=1, delta=1e-5, q_test=1, dis_metric='cosine', rdp=False, eps=-1):
  
  N = len(y_train_few)
  sv = np.zeros(N)

  n_test = len(y_val_few)
  n_test_sub = int(n_test*q_test)
  test_ind = np.random.choice(range(n_test), size=n_test_sub, replace=False)
  x_val_few, y_val_few = x_val_few[test_ind], y_val_few[test_ind]
  n_compose = n_test_sub

  # If eps is specified, find sigma
  if eps>0 and sigma<0:
    sigma = find_noise_multiplier(sampling_probability=q, num_steps=n_compose, target_epsilon=eps, target_delta=delta, eps_error=1e-3, mu_max=5000)
    sigma = sigma / (K*(K+1))
    print('sigma={}'.format(sigma))
  elif eps<0:
    mech = PrivateKNN_SV_RJ_mech(1, sigma, n_compose, K)
    eps = mech.get_approxDP(delta=delta)

    if rdp or eps < 0.01 or eps > 30:
      mech = PrivateKNN_SV_RJ_mech(1, sigma, n_compose, K)
      eps = mech.get_approxDP(delta=delta)
    else:
      prv = PoissonSubsampledGaussianMechanism(sampling_probability=1, noise_multiplier=sigma * (K*(K+1)) )
      acct = PRVAccountant(prvs=prv, max_self_compositions=n_compose, eps_error=1e-3, delta_error=1e-10)
      low, est, upp = acct.compute_epsilon(delta=delta, num_self_compositions=[n_compose])
      eps = upp
  else:
    pass

  for i in tqdm(range(n_test_sub)):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv += private_knn_shapley_RJ_single(x_train_few, y_train_few, x_test, y_test, K, sigma, dis_metric=dis_metric)

  print(sv)
  print(np.argsort(sv))

  return sv, eps, sigma



# x_test, y_test are single data point
def private_knn_shapley_RJ_withsub_single(x_train_few, y_train_few, x_test, y_test, K, sigma, q, dis_metric='cosine'):

  N = len(y_train_few)
  sv = np.zeros(N)

  for l in range(N):

    # Poisson Subsampling
    sub_ind_bool = (np.random.choice([0, 1], size=N, p=[1-q, q])).astype(bool)
    sub_ind_bool[l] = True
    sub_ind = np.where(sub_ind_bool)[0]

    x_train_few_sub, y_train_few_sub = x_train_few[sub_ind], y_train_few[sub_ind]

    N_sub = len(sub_ind)
    sv_temp = np.zeros(N_sub)

    rank = rank_neighbor(x_test, x_train_few_sub, dis_metric=dis_metric)

    sv_temp[int(rank[-1])] += int(y_test==y_train_few_sub[int(rank[-1])]) / N_sub

    for j in range(2, N_sub+1):
      i = N_sub+1-j
      sv_temp[int(rank[-j])] = sv_temp[int(rank[-(j-1)])] + ( (int(y_test==y_train_few_sub[int(rank[-j])]) - int(y_test==y_train_few_sub[int(rank[-(j-1)])])) / K ) * min(K, i) / i
      if sub_ind[ rank[-j] ] == l:
        break
    sv[l] = sv_temp[int(rank[-j])] + np.random.normal(scale=sigma)

  return sv


def private_knn_shapley_RJ_withsub(x_train_few, y_train_few, x_val_few, y_val_few, K, sigma=0, q=1, delta=1e-5, q_test=1, dis_metric='cosine', rdp=False, eps=-1):
  
  N = len(y_train_few)
  sv = np.zeros(N)

  n_test = len(y_val_few)
  n_test_sub = int(n_test*q_test)
  test_ind = np.random.choice(range(n_test), size=n_test_sub, replace=False)
  x_val_few, y_val_few = x_val_few[test_ind], y_val_few[test_ind]

  n_compose = n_test_sub

  # If eps is specified, find sigma
  if eps>0 and sigma<0:
    sigma = find_noise_multiplier(sampling_probability=q, num_steps=n_compose, target_epsilon=eps, target_delta=delta, eps_error=1e-2, mu_max=5000)
    sigma = sigma / (K*(K+1))
    print('sigma={}'.format(sigma))
  elif eps<0:
    mech = PrivateKNN_SV_RJ_mech(q, sigma, n_compose, K)
    eps = mech.get_approxDP(delta=delta)

    if rdp or eps < 0.01 or eps > 30:
      mech = PrivateKNN_SV_RJ_mech(q, sigma, n_compose, K)
      eps = mech.get_approxDP(delta=delta)
    else:
      prv = PoissonSubsampledGaussianMechanism(sampling_probability=q, noise_multiplier=sigma * (K*(K+1)) )
      acct = PRVAccountant(prvs=prv, max_self_compositions=n_compose, eps_error=1e-3, delta_error=1e-10)
      low, est, upp = acct.compute_epsilon(delta=delta, num_self_compositions=[n_compose])
      eps = upp
  else:
    pass

  for i in tqdm(range(n_test_sub)):
    x_test, y_test = x_val_few[i], y_val_few[i]
    sv += private_knn_shapley_RJ_withsub_single(x_train_few, y_train_few, x_test, y_test, K, sigma, q=q, dis_metric=dis_metric)

  return sv, eps, sigma

