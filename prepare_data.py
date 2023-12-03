import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from numpy import linalg as LA
import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset
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

OpenML_dataset = ['fraud', 'apsfail', 'click', 'phoneme', 'wind', 'pol', 'creditcard', 'cpu', 'vehicle', '2dplanes']

# If noisy_data=False => Flip Label
# If noisy_data=True => Add Gaussian Noise
def get_processed_data(dataset, n_data, n_val, flip_ratio, minor_ratio=0.5, noisy_data=False):
    
    np.random.seed(999)
    
    print('-------')
    print('Load Dataset {}'.format(dataset))

    n_flip = int(n_data*flip_ratio)

    if dataset in OpenML_dataset:
        X, y, _, _ = get_data(dataset)
        x_val, y_val = X[:n_val], y[:n_val]
        x_train, y_train = X[n_val:], y[n_val:]

        # Make a balanced dataset
        p = np.mean(y_train)
        if p < 0.5:
            minor_class=1
        else:
            minor_class=0
        index_minor_class = np.where(y_train == minor_class)[0]
        index_major_class = np.where(y_train == 1-minor_class)[0]
        n_minor = int(n_data*minor_ratio)
        n_major = int(n_data - n_minor)
        idx_minor = np.random.choice(index_minor_class, size=n_minor, replace=False)
        idx_major = np.random.choice(index_major_class, size=n_major, replace=False)
        idx = np.concatenate([idx_minor, idx_major])
        idx = np.random.permutation(idx)
        x_train, y_train = x_train[idx], y_train[idx]

        train_l2_norm = LA.norm(x_train, axis=0)
        # print('Average Norm of Data Point', np.mean(train_l2_norm))

        if noisy_data:
            x_train[:n_flip] += np.random.normal( loc=0.0, scale=np.tile(train_l2_norm*1, (n_flip, 1)) )

        # Normalize Dataset
        X_mean, X_std = np.mean(x_train, 0), np.std(x_train, 0)
        normalizer_fn = lambda x: (x - X_mean) / np.clip(X_std, 1e-12, None)
        x_train, x_val = normalizer_fn(x_train), normalizer_fn(x_val)
        train_l2_norm = LA.norm(x_train, axis=1)
        val_l2_norm = LA.norm(x_val, axis=1)
        x_train = x_train / train_l2_norm[:, np.newaxis]
        x_val = x_val / val_l2_norm[:, np.newaxis]

    elif dataset == 'cifar10':
        path ='/home/yq/neural_collapse/features/'
        train_feature = np.load(path + 'vit_cifar_train.npy')
        test_feature = np.load(path + 'vit_cifar_test.npy')
        train_mean = np.mean(train_feature, axis=0)
        #print(f'train_mean is {train_mean}')
        train_var = np.var(train_feature, axis=0)
        test_mean = np.mean(test_feature, axis=0)
        test_var = np.var(test_feature, axis=0)
        #train_feature_center = train_feature -train_mean
        #test_feature_center = test_feature - test_mean
        train_feature_center = (train_feature - train_mean) / np.sqrt(train_var + 1e-5)
        test_feature_center = (test_feature - train_mean) / np.sqrt(train_var + 1e-5)
        train_l2_norm = LA.norm(train_feature_center, axis=1)
        test_l2_norm = LA.norm(test_feature_center, axis=1)
        train_feature_norm = train_feature_center / train_l2_norm[:, np.newaxis]
        test_feature_norm = test_feature_center / test_l2_norm[:, np.newaxis]
        print(f'test the first feature norm is {LA.norm(train_feature_norm[0,:])}')
        train_ds, test_ds = load_dataset('cifar10', split=['train[:]', 'test[:]'])
        train_labels = train_ds['label']
        test_labels = test_ds['label']
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)
        x_train, y_train = make_balance_sample_multiclass(train_feature_norm, train_labels, n_data)
        x_val, y_val = make_balance_sample_multiclass(test_feature_norm, test_labels, n_val)

    elif dataset == 'Gaussian':

        dim = 10        # Set the dimensionality of the Gaussian distribution
        mean = np.zeros(dim)  # Set the mean of the Gaussian distribution
        stddev = np.ones(dim) # Set the standard deviation of the Gaussian distribution

        data = np.random.normal(mean, stddev, (n_data+n_val, dim))
        feature_sum = np.sum(data, axis=1)
        labels = np.where(feature_sum > 0, 1, 0)

        x_train, y_train = data[:n_data], labels[:n_data]
        x_val, y_val = data[n_data:], labels[n_data:]

    else:
        x_train, y_train, x_test, y_test = get_data(dataset)
        x_val, y_val = x_test, y_test

        if dataset != 'covertype':
            x_train, y_train = make_balance_sample_multiclass(x_train, y_train, n_data)
            x_val, y_val = make_balance_sample_multiclass(x_test, y_test, n_data)

    assert len(y_train.shape)==1
    n_class = len(np.unique(y_train))
    print('# of classes = {}'.format(n_class))
    print('-------')

    if noisy_data:
        print('Data Error Type: Noisy Feature')
        print('-------')
        # x_train[:n_flip] += np.random.normal(loc=0.0, scale=100.0, size=(n_flip, x_train.shape[1]))
    else:
        print('Data Error Type: Mislabel')
        print('-------')
        if n_class == 2:
            y_train[:n_flip] = 1 - y_train[:n_flip]
        else:
            y_train[:n_flip] = np.array( [ np.random.choice( np.setdiff1d(np.arange(n_class), [y_train[i]]) ) for i in range(n_flip) ] )

    return x_train, y_train, x_val, y_val



def get_processed_data_clip(dataset, n_data, n_val, flip_ratio, minor_ratio=0.5):
    
    print('-------')
    print('Load Dataset {}'.format(dataset))

    if dataset in OpenML_dataset:

        X, y, _, _ = get_data(dataset)
        x_train, y_train = X[:n_data], y[:n_data]
        x_val, y_val = X[n_data:n_data+n_val], y[n_data:n_data+n_val]

        X_mean, X_std = np.mean(x_train, 0), np.std(x_train, 0)
        normalizer_fn = lambda x: (x - X_mean) / np.clip(X_std, 1e-12, None)
        x_train, x_val = normalizer_fn(x_train), normalizer_fn(x_val)

        x_train = x_train / np.linalg.norm(x_train, ord=2, axis=1)[:, None] * 3
        x_val = x_val / np.linalg.norm(x_val, ord=2, axis=1)[:, None] * 3

    else:
        x_train, y_train, x_test, y_test = get_data(dataset)
        x_val, y_val = x_test, y_test

        if dataset != 'covertype':
            x_train, y_train = make_balance_sample_multiclass(x_train, y_train, n_data)
            x_val, y_val = make_balance_sample_multiclass(x_test, y_test, n_data)

        # x_train = x_train / np.linalg.norm(x_train, ord=2, axis=1)[:, None, None, None] * 3
        # x_val = x_val / np.linalg.norm(x_val, ord=2, axis=1)[:, None, None, None] * 3

    np.random.seed(999)
    n_flip = int(n_data*flip_ratio)

    assert len(y_train.shape)==1
    n_class = len(np.unique(y_train))
    print('# of classes = {}'.format(n_class))
    print('-------')

    if n_class == 2:
        y_train[:n_flip] = 1 - y_train[:n_flip]
    else:
        y_train[:n_flip] = np.array( [ np.random.choice( np.setdiff1d(np.arange(n_class), [y_train[i]]) ) for i in range(n_flip) ] )

    return x_train, y_train, x_val, y_val




def get_processed_data_noisy(dataset, n_data, n_val, flip_ratio, minor_ratio=0.5):
    
    np.random.seed(999)
    
    print('-------')
    print('Load Dataset {}'.format(dataset))

    if dataset in OpenML_dataset:
        X, y, _, _ = get_data(dataset)
        x_val, y_val = X[:n_val], y[:n_val]

        x_train, y_train = X[n_val:], y[n_val:]
        X_mean, X_std = np.mean(x_train, 0), np.std(x_train, 0)
        normalizer_fn = lambda x: (x - X_mean) / np.clip(X_std, 1e-12, None)
        x_train, x_val = normalizer_fn(x_train), normalizer_fn(x_val)

        p = np.mean(y_train)
        if p < 0.5:
            minor_class=1
        else:
            minor_class=0
        
        index_minor_class = np.where(y_train == minor_class)[0]
        index_major_class = np.where(y_train == 1-minor_class)[0]
        n_minor = int(n_data*minor_ratio)
        n_major = int(n_data - n_minor)
        idx_minor = np.random.choice(index_minor_class, size=n_minor, replace=False)
        idx_major = np.random.choice(index_major_class, size=n_major, replace=False)
        idx = np.concatenate([idx_minor, idx_major])
        x_train, y_train = x_train[idx], y_train[idx]

    else:
        x_train, y_train, x_test, y_test = get_data(dataset)
        x_val, y_val = x_test, y_test

        if dataset != 'covertype':
            x_train, y_train = make_balance_sample_multiclass(x_train, y_train, n_data)
            x_val, y_val = make_balance_sample_multiclass(x_test, y_test, n_data)

    n_flip = int(n_data*flip_ratio)

    print(x_train.shape)
    
    x_train[:n_flip] += np.random.normal(loc=10.0, scale=0.0, size=(n_flip, x_train.shape[1]))

    return x_train, y_train, x_val, y_val





def get_data(dataset):

    if dataset in ['covertype']+OpenML_dataset:
        ret = get_minidata(dataset)
    elif dataset == 'MNIST':
        ret = get_mnist()
    elif dataset == 'CIFAR10':
        ret = get_cifar_improved()
    elif dataset == 'Dog_vs_Cat':
        ret = get_dogcat()
    elif dataset == 'Dog_vs_CatFeature':
        ret = get_dogcatFeature()
    elif dataset == 'FMNIST':
        ret = get_fmnist()
    else:
        sys.exit(1)

    return ret


def make_balance_sample(data, target):
    p = np.mean(target)
    if p < 0.5:
        minor_class=1
    else:
        minor_class=0
    
    index_minor_class = np.where(target == minor_class)[0]
    n_minor_class=len(index_minor_class)
    n_major_class=len(target)-n_minor_class
    new_minor=np.random.choice(index_minor_class, size=n_major_class-n_minor_class, replace=True)

    data=np.concatenate([data, data[new_minor]])
    target=np.concatenate([target, target[new_minor]])
    return data, target


def get_minidata(dataset):

    open_ml_path = 'OpenML_datasets/'

    np.random.seed(999)

    if dataset == 'covertype':
        x_train, y_train, x_test, y_test = pickle.load( open('covertype_200.dataset', 'rb') )

    elif dataset == 'fraud':
        data_dict=pickle.load(open(open_ml_path+'CreditCardFraudDetection_42397.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    elif dataset == 'apsfail':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'APSFailure_41138.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    elif dataset == 'click':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'Click_prediction_small_1218.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    elif dataset == 'phoneme':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'phoneme_1489.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    elif dataset == 'wind':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'wind_847.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    elif dataset == 'pol':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'pol_722.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    elif dataset == 'creditcard':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'default-of-credit-card-clients_42477.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    elif dataset == 'cpu':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'cpu_act_761.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    elif dataset == 'vehicle':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'vehicle_sensIT_357.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    elif dataset == '2dplanes':
        # ((10000, 28), array([0., 1.], dtype=float32))
        data_dict=pickle.load(open(open_ml_path+'2dplanes_727.pkl', 'rb'))
        data, target = data_dict['X_num'], data_dict['y'] # data[:,1:], data[:,0]
        target = (target == 1) + 0.0
        target = target.astype(np.int32)
        data, target=make_balance_sample(data, target)

    else:
        print('No such dataset!')
        sys.exit(1)


    if dataset not in ['covertype']:
        idxs=np.random.permutation(len(data))
        data, target=data[idxs], target[idxs]
        return data, target, None, None
    else:
        return x_train, y_train, x_test, y_test



def get_mnist():
    transform_train = transforms.Compose([transforms.ToTensor(),])
    transform_test = transforms.Compose([transforms.ToTensor(),])
    trainset = datasets.MNIST(root='.', train=True, download=True,transform=transform_train)
    testset = datasets.MNIST(root='.', train=False, download=True,transform=transform_test)

    (x_train, y_train), (x_test, y_test) = (trainset.data, trainset.targets), (testset.data, testset.targets)

    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train.numpy()
    y_train = y_train.numpy()
    x_test = x_test.numpy()
    y_test = y_test.numpy()

    return x_train, y_train, x_test, y_test



def get_fmnist():
    transform_train = transforms.Compose([transforms.ToTensor(),])
    transform_test = transforms.Compose([transforms.ToTensor(),])
    trainset = datasets.FashionMNIST(root='.', train=True, download=True,transform=transform_train)
    testset = datasets.FashionMNIST(root='.', train=False, download=True,transform=transform_test)

    (x_train, y_train), (x_test, y_test) = (trainset.data, trainset.targets), (testset.data, testset.targets)

    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train.numpy()
    y_train = y_train.numpy()
    x_test = x_test.numpy()
    y_test = y_test.numpy()

    return x_train, y_train, x_test, y_test



def get_cifar_improved():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='.', train=True, download=True, transform=transform_train) 
    testset = torchvision.datasets.CIFAR10(root='.', train=False, download=True, transform=transform_test)

    return trainset, testset


def get_cifar():
    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    trainset = torchvision.datasets.CIFAR10(root='.', train=True, download=True, transform=transform_train) 
    testset = torchvision.datasets.CIFAR10(root='.', train=False, download=True, transform=transform_test)

    (x_train, y_train), (x_test, y_test) = (trainset.data, trainset.targets), (testset.data, testset.targets)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test


def get_dogcat():

    x_train, y_train, x_test, y_test = get_cifar()

    dogcat_ind = np.where(np.logical_or(y_train==3, y_train==5))[0]
    x_train, y_train = x_train[dogcat_ind], y_train[dogcat_ind]
    y_train[y_train==3] = 0
    y_train[y_train==5] = 1

    dogcat_ind = np.where(np.logical_or(y_test==3, y_test==5))[0]
    x_test, y_test = x_test[dogcat_ind], y_test[dogcat_ind]
    y_test[y_test==3] = 0
    y_test[y_test==5] = 1

    return x_train, y_train, x_test, y_test


def get_dogcatFeature():

    # x_train, y_train, x_test, y_test = pickle.load( open('dogvscat_feature.dataset', 'rb') )
    x_train, y_train, x_test, y_test = pickle.load( open('result/DogCatImageNetPretrain.data', 'rb') )

    return x_train, y_train, x_test, y_test



