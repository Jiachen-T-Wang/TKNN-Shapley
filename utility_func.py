import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms
from torchvision import models

# general
import pandas as pd 
import numpy as np 
import copy
import pickle
import sys
import time
import os

from random import randint

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from helper import *
from models import *
from utils import *
import pdb


def set_parameter_requires_grad(model, feature_extracting):
  if feature_extracting:
    for param in model.parameters():
      param.requires_grad = True



def GENERAL_data_to_acc_loss(dataset, model_type, trainset, testset, idx, args, weights=None):

  # Set Random Seed for Initialization
  if hasattr(args, 'init_random_state'):
    torch.manual_seed(args.init_random_state)

  # Set training hyperpamameters
  batch_size, lr = args.batch_size, args.lr

  if dataset in ['CIFAR10']:

    if model_type[:3] == 'VGG':
      net = VGG(model_type).cuda()
    elif model_type == 'ResNet18':
      net = ResNet18().cuda()
    elif model_type == 'ResNet50':
      net = ResNet50().cuda()
    elif model_type == 'DenseNet':
      net = densenet_cifar().cuda()
    elif model_type == 'SmallCNN':
      net = SmallCNN_CIFAR().cuda()
    elif model_type == 'ResNet18_pretrained':

      net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
      set_parameter_requires_grad(net, feature_extracting=True)
      net.fc = nn.Linear(net.fc.in_features, 10)
      input_size = 224
      transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])
      transform_test = transforms.Compose([
        transforms.Resize(input_size),
        # transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])
      trainset = torchvision.datasets.CIFAR10(root='.', train=True, download=True, transform=transform_train) 
      testset = torchvision.datasets.CIFAR10(root='.', train=False, download=True, transform=transform_test)
      net = net.cuda()

    else:
      print('not supported')

    criterion = nn.CrossEntropyLoss()
    if model_type == 'ResNet18_pretrained':
      # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
      optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
      # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
      scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45, 60])
      n_epoch = 70
    elif model_type in ['ResNet18', 'VGG11']:
      optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
      n_epoch = 200
    else:
      optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-7)
      n_epoch = 20
      scheduler = None

  elif dataset in ['MNIST', 'FMNIST']:

    if model_type == 'Logistic':
      net = MnistLogistic().cuda()
    elif model_type == 'SmallCNN':
      net = MnistLeNet().cuda()
    elif model_type == 'LargeCNN':
      net = MnistLargeCNN().cuda()
    else:
      print('not supported')

    n_epoch = 50
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-7)
    scheduler = None


  # Set Random Seed for Batch Selection
  if hasattr(args, 'batch_random_state'):
    torch.manual_seed(args.batch_random_state)

  trainset = torch.utils.data.Subset(trainset, idx)

  """
  if args.withreplace: 
    weights = np.ones(len(trainset))
  """

  if weights is None:
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=1)
  else:
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=batch_size, replacement=True)
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=False, sampler=sampler)

  test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

  def train(epoch):
      print('\nEpoch: %d' % epoch)
      net.train()
      train_loss = 0
      correct = 0
      total = 0
      for batch_idx, (inputs, targets) in enumerate(train_loader):
          inputs = Variable(inputs)
          targets = Variable(targets).long()
          inputs, targets = inputs.cuda(), targets.cuda()
          optimizer.zero_grad()
          outputs = net(inputs)
          loss = criterion(outputs, targets)
          loss.backward()
          optimizer.step()

          train_loss += loss.item()
          _, predicted = outputs.max(1)
          total += targets.size(0)
          correct += predicted.eq(targets).sum().item()
          progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

      acc = correct/total
      return train_loss, acc

  def test(epoch):
      global best_acc
      net.eval()
      test_loss = 0
      correct = 0
      total = 0
      with torch.no_grad():
          for batch_idx, (inputs, targets) in enumerate(test_loader):
              inputs = Variable(inputs)
              targets = Variable(targets).long()
              inputs, targets = inputs.cuda(), targets.cuda()
              outputs = net(inputs)
              loss = criterion(outputs, targets)
              test_loss += loss.item()
              _, predicted = outputs.max(1)
              total += targets.size(0)
              correct += predicted.eq(targets).sum().item()
              progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
      acc = correct/total
      return test_loss, acc

  loss_lst, acc_lst = [], []
  for e in range(n_epoch):
    train_loss, train_acc = train(e)
    test_loss, test_acc = test(e)
    if scheduler != None: scheduler.step()
    print("Epoch: {}. || Train_Loss: {}, Train_Acc: {} || Val_Loss: {}. Val_Acc: {}".format(e, train_loss, train_acc, test_loss, test_acc))
    
    loss_lst.append(test_loss)
    acc_lst.append(test_acc)

  return [acc_lst, loss_lst]






def torch_general_data_to_net(dataset, model_type, x_train, y_train, x_test, y_test, weights=None, verbose=0, batch_size=32, lr=0.001):

  # Process Datasets and Initialize Models
  if dataset in ['MNIST', 'FMNIST']:

    if len(y_train) == 0: return 0.1

    if x_train.shape[1]==28:
      x_train = np.moveaxis(x_train, 3, 1)
      x_test = np.moveaxis(x_test, 3, 1)

    if len(y_train.shape)>1 and y_train.shape[1]>1: y_train = np.argmax(y_train, axis=1)
    
    if len(y_test.shape)>1 and y_test.shape[1]>1: y_test = np.argmax(y_test, axis=1)

    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    if model_type == 'Logistic':
      net = MnistLogistic().cuda()
    elif model_type == 'SmallCNN':
      net = MnistLeNet().cuda()
    elif model_type == 'LargeCNN':
      net = MnistLargeCNN().cuda()

    criterion = torch.nn.CrossEntropyLoss()
    n_epoch = 50
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-7)


  elif dataset == 'Dog_vs_CatFeature':

    factor = 1

    if model_type == 'Logistic':
      net = DogCatLogistic().cuda()
    elif model_type == 'MLP':
      net = DogCatMLP(factor).cuda()
    elif model_type == 'MLP_RS':
      net = DogCatMLP(factor).cuda()
    else:
      print('not supported')

    criterion = torch.nn.CrossEntropyLoss()
    n_epoch = 50
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-7)

  else:
    print('DATASET NOT SUPPORTED')
    exit(1)


  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)

  if weights is None:
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=True)
  else:
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=len(y_train))
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  # Training
  acc_lst = []
  for epoch in range(n_epoch):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).long()
          optimizer.zero_grad()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
      correct = 0
      total = 0
      for images, labels in test_loader:
          images = Variable(images)
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()
      accuracy = correct/total

      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))
      acc_lst.append(accuracy.item())
      
  return net





def torch_general_data_to_acclst(dataset, model_type, x_train, y_train, x_test, y_test, weights=None, verbose=0, batch_size=32, lr=0.001):

  # Process Datasets and Initialize Models
  if dataset in ['MNIST', 'FMNIST']:

    if len(y_train) == 0: return 0.1

    if x_train.shape[1]==28:
      x_train = np.moveaxis(x_train, 3, 1)
      x_test = np.moveaxis(x_test, 3, 1)

    if len(y_train.shape)>1 and y_train.shape[1]>1: y_train = np.argmax(y_train, axis=1)
    
    if len(y_test.shape)>1 and y_test.shape[1]>1: y_test = np.argmax(y_test, axis=1)

    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    if model_type == 'Logistic':
      net = MnistLogistic().cuda()
      n_epoch = 30
    elif model_type == 'SmallCNN':
      net = MnistLeNet().cuda()
      n_epoch = 50
    elif model_type == 'LargeCNN':
      net = MnistLargeCNN().cuda()
      n_epoch = 50

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-7)


  elif dataset == 'Dog_vs_CatFeature':

    factor = 1

    if model_type == 'Logistic':
      net = DogCatLogistic().cuda()
      n_epoch = 30
    elif model_type == 'MLP':
      net = DogCatMLP(factor).cuda()
      n_epoch = 50
    elif model_type == 'MLP_RS':
      net = DogCatMLP(factor).cuda()
      n_epoch = 50
    else:
      print('not supported')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-7)

  else:
    print('DATASET NOT SUPPORTED')
    exit(1)


  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)

  if weights is None:
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=True)
  else:
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=batch_size)
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


  # Training
  acc_lst = []

  for epoch in range(n_epoch):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).long()
          optimizer.zero_grad()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
      correct = 0
      total = 0
      for images, labels in test_loader:
          images = Variable(images)
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()
      accuracy = correct/total

      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))

      acc_lst.append(accuracy.item())

  return acc_lst









def torch_mnist_data_to_acc(model_type, x_train, y_train, x_test, y_test, weights=None, verbose=0, batch_size=32, lr=0.001):

  if len(y_train) == 0: return 0.1

  if x_train.shape[1]==28:
    x_train = np.moveaxis(x_train, 3, 1)
    x_test = np.moveaxis(x_test, 3, 1)

  if len(y_train.shape)>1 and y_train.shape[1]>1:
    y_train = np.argmax(y_train, axis=1)
  
  if len(y_test.shape)>1 and y_test.shape[1]>1:
    y_test = np.argmax(y_test, axis=1)

  y_train = y_train.reshape(-1)
  y_test = y_test.reshape(-1)

  criterion = torch.nn.CrossEntropyLoss()

  if model_type == 'Logistic':
    net = MnistLogistic().cuda()
  elif model_type == 'SmallCNN':
    net = MnistLeNet().cuda()
  elif model_type == 'LargeCNN':
    net = MnistLargeCNN().cuda()

  optimizer = torch.optim.Adam(net.parameters(), lr=lr)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)

  if weights is None:
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=True)
  else:
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=batch_size)
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  acc_lst = []

  for epoch in range(30):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).long()
          optimizer.zero_grad()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
      correct = 0
      total = 0
      for images, labels in test_loader:
          images = Variable(images)
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()
      accuracy = correct/total
      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))

      acc_lst.append(accuracy.item())

  acc = get_converge(acc_lst, patience=3, loss=False)

  return acc




def torch_cifar_data_to_acc(model_type, x_train, y_train, x_test, y_test, weights=None, verbose=0, batch_size=32, lr=0.001):

  if x_train.shape[1]==32:
    x_train = np.moveaxis(x_train, 3, 1)
    x_test = np.moveaxis(x_test, 3, 1)

  y_train = y_train.reshape(-1)
  y_test = y_test.reshape(-1)

  criterion = torch.nn.CrossEntropyLoss()

  if model_type[:3] == 'VGG':
    net = VGG(model_type).cuda()
  elif model_type == 'ResNet18':
    net = ResNet18().cuda()
  elif model_type == 'ResNet50':
    net = ResNet50().cuda()
  elif model_type == 'DenseNet':
    net = densenet_cifar().cuda()
  elif model_type == 'SmallCNN':
    net = SmallCNN_CIFAR().cuda()
  else:
    print('not supported')

  n_epoch = 50

  optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-7)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)

  if weights is None:
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=True)
  else:
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=batch_size)
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  max_acc = 0

  for epoch in range(30):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).long()
          optimizer.zero_grad()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
      correct = 0
      total = 0
      for images, labels in test_loader:
          images = Variable(images)
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()
      accuracy = correct/total
      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))

      max_acc = max(max_acc, accuracy.item())
  return max_acc




def torch_general_data_to_acc_time_loss(dataset, model_type, x_train, y_train, x_test, y_test, args, weights=None, verbose=0):

  # Set Random Seed for Initialization
  torch.manual_seed(args.init_random_state)

  criterion = torch.nn.CrossEntropyLoss()
  metric = torch.nn.CrossEntropyLoss(reduction='sum')

  # Process Datasets and Initialize Models
  if dataset in ['MNIST', 'FMNIST']:

    if len(y_train) == 0: return 0.1

    if x_train.shape[1]==28:
      x_train = np.moveaxis(x_train, 3, 1)
      x_test = np.moveaxis(x_test, 3, 1)

    if len(y_train.shape)>1 and y_train.shape[1]>1: y_train = np.argmax(y_train, axis=1)
    
    if len(y_test.shape)>1 and y_test.shape[1]>1: y_test = np.argmax(y_test, axis=1)

    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    if model_type == 'Logistic':
      net = MnistLogistic().cuda()
    elif model_type == 'SmallCNN':
      net = MnistLeNet().cuda()
    elif model_type == 'LargeCNN':
      net = MnistLargeCNN().cuda()

    criterion = torch.nn.CrossEntropyLoss()
    n_epoch = 50

  elif dataset == 'Dog_vs_CatFeature':

    factor = 1

    if model_type == 'Logistic':
      net = DogCatLogistic().cuda()
    elif model_type == 'MLP':
      net = DogCatMLP(factor).cuda()
    elif model_type == 'MLP_RS':
      net = DogCatMLP(factor).cuda()
    else:
      print('not supported')

    criterion = torch.nn.CrossEntropyLoss()
    n_epoch = 50

  elif dataset == 'CIFAR10':
    
    factor = 1
    
    if x_train.shape[1]==32:
      x_train = np.moveaxis(x_train, 3, 1)
      x_test = np.moveaxis(x_test, 3, 1)
      y_train = y_train.reshape(-1)
      y_test = y_test.reshape(-1)
    
    if model_type[:3] == 'VGG':
      net = VGG(model_type).cuda()
    elif model_type == 'ResNet18':
      net = ResNet18().cuda()
    elif model_type == 'ResNet50':
      net = ResNet50().cuda()
    elif model_type == 'DenseNet':
      net = densenet_cifar().cuda()
    elif model_type == 'SmallCNN':
      net = SmallCNN_CIFAR().cuda()
    else:
      print('not supported')
      
    criterion = torch.nn.CrossEntropyLoss()
    n_epoch = 100

  else:
    print('DATASET NOT SUPPORTED')
    exit(1)

  s = time.time()
  time_lst = []

  # Scale the initialized parameter
  net = scale_init_param(net, factor=args.init_scale)

  time_lst.append(time.time()-s)
  
  # pdb.set_trace()

  # Set trainig hyperpamameters
  batch_size, lr = args.batch_size, args.lr
  optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-7)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)

  # Set Random Seed for Batch Selection
  torch.manual_seed(args.batch_random_state)
  
  if args.withreplace:
    weights = np.ones(len(y_train))

  if weights is None:
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=True)
  else:
    print('withreplacement')
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=batch_size, replacement=True)
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  acc_lst = []
  loss_lst = []

  for epoch in range(n_epoch):
    
      tr_correct, tr_total = 0, 0
      tr_loss = 0

      s = time.time()

      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).long()
          optimizer.zero_grad()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          _, predicted = torch.max(outputs.data, 1)
          tr_total += labels.size(0)
          tr_correct += (predicted == labels).sum()
          loss.backward()
          tr_loss = metric(outputs, labels).item()
          optimizer.step()

      tr_acc = tr_correct / tr_total
      tr_loss = tr_loss / tr_total
      
      time_lst.append(time.time()-s)
      # Test Phase
      
      te_correct, te_total = 0, 0
      val_loss =  0

      for images, labels in test_loader:
          images = Variable(images)
          labels = Variable(labels).long()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          _, predicted = torch.max(outputs.data, 1)
          te_total += labels.size(0)
          te_correct += (predicted == labels).sum()
          loss = metric(outputs, labels)
          val_loss += loss.item()

      val_acc = te_correct/te_total
      
      if verbose:
        print("Epoch: {}. || Train_Loss: {}, Train_Acc: {} || Val_Loss: {}. Val_Acc: {}".format(epoch, tr_loss, tr_acc, val_loss, val_acc))

      if args.record_train:
        loss_lst.append(tr_loss)
        acc_lst.append(tr_acc.item())
      else:
        loss_lst.append(val_loss)
        acc_lst.append(val_acc.item())

  print(acc_lst)
  print(loss_lst)
  print(time_lst)

  return [acc_lst, loss_lst, time_lst]



def torch_general_data_to_acc_loss(dataset, model_type, x_train, y_train, x_test, y_test, args, weights=None, verbose=0):

  # Set Random Seed for Initialization
  torch.manual_seed(args.init_random_state)

  criterion = torch.nn.CrossEntropyLoss()
  metric = torch.nn.CrossEntropyLoss(reduction='sum')

  # Set training hyperpamameters
  batch_size, lr = args.batch_size, args.lr

  # Process Datasets and Initialize Models
  if dataset in ['MNIST', 'FMNIST']:

    if len(y_train) == 0: return 0.1

    if x_train.shape[1]==28:
      x_train = np.moveaxis(x_train, 3, 1)
      x_test = np.moveaxis(x_test, 3, 1)

    if len(y_train.shape)>1 and y_train.shape[1]>1: y_train = np.argmax(y_train, axis=1)
    
    if len(y_test.shape)>1 and y_test.shape[1]>1: y_test = np.argmax(y_test, axis=1)

    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    if model_type == 'Logistic':
      net = MnistLogistic().cuda()
    elif model_type == 'SmallCNN':
      net = MnistLeNet().cuda()
    elif model_type == 'LargeCNN':
      net = MnistLargeCNN().cuda()

    criterion = torch.nn.CrossEntropyLoss()
    n_epoch = 50
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-7)


  elif dataset == 'Dog_vs_CatFeature':

    factor = 1

    if model_type == 'Logistic':
      net = DogCatLogistic().cuda()
    elif model_type == 'MLP':
      net = DogCatMLP(factor).cuda()
    elif model_type == 'MLP_RS':
      net = DogCatMLP(factor).cuda()
    else:
      print('not supported')

    criterion = torch.nn.CrossEntropyLoss()
    n_epoch = 50
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-7)


  elif dataset == 'CIFAR10':
    
    factor = 1
    
    if x_train.shape[1]==32:
      x_train = np.moveaxis(x_train, 3, 1)
      x_test = np.moveaxis(x_test, 3, 1)
      y_train = y_train.reshape(-1)
      y_test = y_test.reshape(-1)
    
    if model_type[:3] == 'VGG':
      net = VGG(model_type).cuda()
    elif model_type == 'ResNet18':
      net = ResNet18().cuda()
    elif model_type == 'ResNet50':
      net = ResNet50().cuda()
    elif model_type == 'DenseNet':
      net = densenet_cifar().cuda()
    elif model_type == 'SmallCNN':
      net = SmallCNN_CIFAR().cuda()
    else:
      print('not supported')
      
    criterion = torch.nn.CrossEntropyLoss()
    n_epoch = 200
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])

  else:
    print('DATASET NOT SUPPORTED')
    exit(1)

  # Scale the initialized parameter
  # net = scale_init_param(net, factor=args.init_scale)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)

  # Set Random Seed for Batch Selection
  torch.manual_seed(args.batch_random_state)
  
  if args.withreplace:
    weights = np.ones(len(y_train))

  if weights is None:
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=True)
  else:
    print('withreplacement')
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=batch_size, replacement=True)
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=100)

  acc_lst = []
  loss_lst = []

  for epoch in range(n_epoch):
    
      tr_correct, tr_total = 0, 0
      tr_loss = 0

      net.train()
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).long()
          optimizer.zero_grad()
          outputs = net(images)
          # outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          _, predicted = torch.max(outputs.data, 1)
          tr_total += labels.size(0)
          tr_correct += (predicted == labels).sum()
          loss.backward()
          tr_loss = metric(outputs, labels).item()
          optimizer.step()

      scheduler.step()

      tr_acc = tr_correct / tr_total
      tr_loss = tr_loss / tr_total
      
      # Test Phase
      te_correct, te_total = 0, 0
      val_loss =  0

      net.eval()
      with torch.no_grad():
        for images, labels in test_loader:
            images = Variable(images)
            labels = Variable(labels).long()
            outputs = net(images)
            # outputs = F.softmax(logits, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            te_total += labels.size(0)
            te_correct += (predicted == labels).sum()
            loss = metric(outputs, labels)
            val_loss += loss.item()

      val_acc = te_correct/te_total
      
      if verbose:
        print("Epoch: {}. || Train_Loss: {}, Train_Acc: {} || Val_Loss: {}. Val_Acc: {}".format(epoch, tr_loss, tr_acc, val_loss, val_acc))

      if args.record_train:
        loss_lst.append(tr_loss)
        acc_lst.append(tr_acc.item())
      else:
        loss_lst.append(val_loss)
        acc_lst.append(val_acc.item())

  print(acc_lst)
  print(loss_lst)

  return [acc_lst, loss_lst]









def torch_mnist_data_to_acc_loss(model_type, x_train, y_train, x_test, y_test, weights=None, verbose=0, batch_size=32, lr=0.001):

  if len(y_train) == 0: return 0.1

  if x_train.shape[1]==28:
    x_train = np.moveaxis(x_train, 3, 1)
    x_test = np.moveaxis(x_test, 3, 1)

  if len(y_train.shape)>1 and y_train.shape[1]>1:
    y_train = np.argmax(y_train, axis=1)
  
  if len(y_test.shape)>1 and y_test.shape[1]>1:
    y_test = np.argmax(y_test, axis=1)

  y_train = y_train.reshape(-1)
  y_test = y_test.reshape(-1)

  criterion = torch.nn.CrossEntropyLoss()
  metric = torch.nn.CrossEntropyLoss(reduction='sum')

  if model_type == 'Logistic':
    net = MnistLogistic().cuda()
  elif model_type == 'SmallCNN':
    net = MnistLeNet().cuda()
  elif model_type == 'LargeCNN':
    net = MnistLargeCNN().cuda()

  optimizer = torch.optim.Adam(net.parameters(), lr=lr)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)

  if weights is None:
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=True)
  else:
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=batch_size)
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  acc_lst = []
  loss_lst = []

  n_epoch = 100

  for epoch in range(n_epoch):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).long()
          optimizer.zero_grad()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
      
      correct = 0
      total = 0
      val_loss =  0

      for images, labels in test_loader:
          images = Variable(images)
          labels = Variable(labels).long()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()

          loss = metric(outputs, labels)
          val_loss += loss.item()

      val_acc = correct/total
      if verbose:
        print("Epoch: {}. Val_Loss: {}. Accuracy: {}.".format(epoch, val_loss, val_acc))

      loss_lst.append(val_loss)
      acc_lst.append(val_acc.item())

  return [acc_lst, loss_lst]




def scale_init_param(net, factor=1):

  for param in net.parameters():
    param.data *= factor

  return net



def torch_cifar_data_to_acc_loss(model_type, x_train, y_train, x_test, y_test, args, weights=None, verbose=0):

  if x_train.shape[1]==32:
    x_train = np.moveaxis(x_train, 3, 1)
    x_test = np.moveaxis(x_test, 3, 1)
  y_train = y_train.reshape(-1)
  y_test = y_test.reshape(-1)

  criterion = torch.nn.CrossEntropyLoss()
  metric = torch.nn.CrossEntropyLoss(reduction='sum')


  # Set Random Seed for Initialization
  torch.manual_seed(args.init_random_state)

  if model_type[:3] == 'VGG':
    net = VGG(model_type).cuda()
  elif model_type == 'ResNet18':
    net = ResNet18().cuda()
  elif model_type == 'ResNet50':
    net = ResNet50().cuda()
  elif model_type == 'DenseNet':
    net = densenet_cifar().cuda()
  elif model_type == 'SmallCNN':
    net = SmallCNN_CIFAR().cuda()
  else:
    print('not supported')

  # Scale the initialized parameter
  net = scale_init_param(net, factor=args.init_scale)

  # Set trainig hyperpamameters
  n_epoch = 100
  batch_size, lr = args.batch_size, args.lr
  optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-7)


  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)

  # Set Random Seed for Batch Selection
  torch.manual_seed(args.batch_random_state)

  if weights is None:
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=True)
  else:
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=batch_size)
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  acc_lst = []
  loss_lst = []

  for epoch in range(n_epoch):
    
      train_correct = 0

      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).long()
          optimizer.zero_grad()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

      correct = 0
      total = 0
      val_loss =  0

      for images, labels in test_loader:
          images = Variable(images)
          labels = Variable(labels).long()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()

          loss = metric(outputs, labels)
          val_loss += loss.item()

      val_acc = correct/total
      if verbose:
        print("Epoch: {}. Val_Loss: {}. Accuracy: {}.".format(epoch, val_loss, val_acc))

      loss_lst.append(val_loss)
      acc_lst.append(val_acc.item())

  return [acc_lst, loss_lst]





def torch_binary_data_to_acc_loss(model_type, x_train, y_train, x_test, y_test, args, weights=None, verbose=0):


  criterion = torch.nn.CrossEntropyLoss()
  metric = torch.nn.CrossEntropyLoss(reduction='sum')

  # Set Random Seed for Initialization
  torch.manual_seed(args.init_random_state)

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  if model_type == 'MLP':
    net = BinaryMLP( x_train.shape[1], max(100, int(x_train.shape[1]/2)) ).to(device)
  elif model_type == 'MLP_RS':
    net = BinaryMLP( x_train.shape[1], max(100, int(x_train.shape[1]/2)) ).to(device)
  else:
    print('not supported')


  # Scale the initialized parameter
  net = scale_init_param(net, factor=args.init_scale)

  # Set trainig hyperpamameters
  n_epoch = 30
  batch_size, lr = args.batch_size, args.lr
  optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-7)


  # Set Random Seed for Batch Selection
  torch.manual_seed(args.batch_random_state)

  tensor_x, tensor_y = torch.Tensor(x_train).to(device), torch.Tensor(y_train).to(device)
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)

  if weights is None:
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=True)
  else:
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=batch_size)
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

  tensor_x_test, tensor_y_test = torch.Tensor(x_test).to(device), torch.Tensor(y_test).to(device)
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  acc_lst = []
  loss_lst = []

  for epoch in range(n_epoch):

      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).long()
          optimizer.zero_grad()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

      correct = 0
      total = 0
      val_loss =  0

      for images, labels in test_loader:
          images = Variable(images)
          labels = Variable(labels).long()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()

          loss = metric(outputs, labels)
          val_loss += loss.item()

      val_acc = correct/total
      if verbose:
        print("Epoch: {}. Val_Loss: {}. Accuracy: {}.".format(epoch, val_loss, val_acc))

      loss_lst.append(val_loss)
      acc_lst.append(val_acc.item())

  return [acc_lst, loss_lst]








def torch_dogcat_data_to_acc(model_type, x_train, y_train, x_test, y_test, weights=None, verbose=0, batch_size=32, lr=0.001, return_net=False):

  if x_train.shape[1]==32:
    x_train = np.moveaxis(x_train, 3, 1)
    x_test = np.moveaxis(x_test, 3, 1)

  y_train = y_train.reshape(-1)
  y_test = y_test.reshape(-1)

  criterion = torch.nn.CrossEntropyLoss()

  if model_type == 'ResNet18':
    net = ResNet18(num_classes=2).cuda()
  else:
    print('not supported')

  n_epoch = 50

  optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-7)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)

  if weights is None:
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=True)
  else:
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=batch_size)
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  max_acc = 0

  for epoch in range(30):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).long()
          optimizer.zero_grad()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
      correct = 0
      total = 0
      for images, labels in test_loader:
          images = Variable(images)
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()
      accuracy = correct/total
      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))

      if accuracy.item() > max_acc:
        net_best = net

      max_acc = max(max_acc, accuracy.item())

  if return_net:
    return max_acc, net_best
  else:
    return max_acc



def torch_dogcatFeature_data_to_acc(model_type, x_train, y_train, x_test, y_test, weights=None, verbose=0, batch_size=32, lr=0.001, return_net=False, factor=1):

  criterion = torch.nn.CrossEntropyLoss()

  if model_type == 'Logistic':
    net = DogCatLogistic().cuda()
  elif model_type == 'MLP':
    net = DogCatMLP(factor).cuda()
  elif model_type == 'MLP_RS':
    net = DogCatMLP(factor).cuda()
  else:
    print('not supported')

  n_epoch = 50

  optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-7)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)

  if weights is None:
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=True)
  else:
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=batch_size)
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  max_acc = 0

  for epoch in range(n_epoch):

      for i, (images, labels) in enumerate(train_loader):

        images = Variable(images)
        labels = Variable(labels).long()

        if model_type == 'MLP_RS':

          with torch.no_grad():
            original_param = [param.data.clone() for param in net.parameters()]

          # Randomly perturb parameters
          for param, o_param in zip(net.parameters(), original_param):
            param.data = o_param + lr * (torch.FloatTensor(param.size()).normal_(0, 2)).cuda()

          optimizer.zero_grad()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          loss.backward()

          # Get back to original gradient
          for param, o_param in zip(net.parameters(), original_param):
            param.data = o_param

          optimizer.step()

        else:

          optimizer.zero_grad()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

      """
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).long()
          optimizer.zero_grad()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
      """

      correct = 0
      total = 0
      for images, labels in test_loader:
          images = Variable(images)
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()
      accuracy = correct/total
      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))

      if accuracy.item() > max_acc:
        net_best = net

      max_acc = max(max_acc, accuracy.item())

  if return_net:
    return max_acc, net_best
  else:
    return max_acc



def binary_data_to_acc(model_type, x_train, y_train, x_test, y_test, w=None):
  if model_type == 'Logistic':
    model = LogisticRegression(max_iter=5000, solver='liblinear')
  elif model_type == 'SVM':
    model = SVC(kernel='rbf', max_iter=5000, C=1)
  if len(y_train)==0:
    return 0.5, 0.5
  try:
    model.fit(x_train, y_train, sample_weight=w)
  except:
    return 0.5
  acc = model.score(x_test, y_test)
  return acc



def torch_binary_data_to_acc(model_type, x_train, y_train, x_test, y_test, weights=None, verbose=0, batch_size=32, lr=0.001, return_net=False):

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  if model_type == 'MLP':
    net = BinaryMLP( x_train.shape[1], max(100, int(x_train.shape[1]/2)) ).to(device)
  elif model_type == 'MLP_RS':
    net = BinaryMLP( x_train.shape[1], max(100, int(x_train.shape[1]/2)) ).to(device)
  else:
    print('not supported')


  n_epoch = 30
  optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-7)
  criterion = torch.nn.CrossEntropyLoss()

  tensor_x, tensor_y = torch.Tensor(x_train).to(device), torch.Tensor(y_train).to(device)
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)

  if weights is None:
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=True)
  else:
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=batch_size)
    train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

  tensor_x_test, tensor_y_test = torch.Tensor(x_test).to(device), torch.Tensor(y_test).to(device)
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  acc_lst = []

  for epoch in range(n_epoch):
      for i, (images, labels) in enumerate(train_loader):

        images = Variable(images)
        labels = Variable(labels).long()

        if model_type == 'MLP_RS':

          with torch.no_grad():
            original_param = [param.data.clone() for param in net.parameters()]

          # Randomly perturb parameters
          for param, o_param in zip(net.parameters(), original_param):
            param.data = o_param + lr * (torch.FloatTensor(param.size()).normal_(0, 1)).to(device)

          optimizer.zero_grad()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          loss.backward()

          # Get back to original gradient
          for param, o_param in zip(net.parameters(), original_param):
            param.data = o_param

          optimizer.step()

        else:

          optimizer.zero_grad()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

      correct = 0
      total = 0
      for images, labels in test_loader:
          images = Variable(images)
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()
      accuracy = correct/total
      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))

      acc_lst.append(accuracy.item())

  acc = get_converge(acc_lst, patience=3, loss=False)

  return acc



def isMonotonic(A, increase):
  if increase:
    return (all(A[i] <= A[i + 1] for i in range(len(A) - 1)))
  else:
    return (all(A[i] >= A[i + 1] for i in range(len(A) - 1)))


def get_converge(array, patience=3, loss=False):

  for i in range(len(array)-patience+1):
    subarray = array[i:i+patience]

    if loss:
      if isMonotonic(subarray, increase=True):
        return array[i]
    else:
      if isMonotonic(subarray, increase=False):
        return array[i]
  
  return array[-1]








"""
def torch_mnist_Logistic_data_to_acc_weighted(x_train, y_train, x_test, y_test, weights, verbose=0, batch_size=8, lr=0.001):

  if len(y_train) == 0:
    return 0.1, None

  if x_train.shape[1]==28:
    x_train = np.moveaxis(x_train, 3, 1)
    x_test = np.moveaxis(x_test, 3, 1)

  if len(y_train.shape)>1 and y_train.shape[1]>1:
    y_train = np.argmax(y_train, axis=1)
  
  if len(y_test.shape)>1 and y_test.shape[1]>1:
    y_test = np.argmax(y_test, axis=1)

  y_train = y_train.reshape(-1)
  y_test = y_test.reshape(-1)

  criterion = torch.nn.CrossEntropyLoss()
  net = MnistLogistic().cuda()
  
  optimizer = torch.optim.Adam(net.parameters(), lr=lr)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)

  sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=batch_size)
  train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)

  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  max_acc = 0

  for epoch in range(30):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).long()
          optimizer.zero_grad()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
      correct = 0
      total = 0
      for images, labels in test_loader:
          images = Variable(images)
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()
      accuracy = correct/total
      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))

      max_acc = max(max_acc, accuracy.item())
  return max_acc


def torch_mnist_Logistic_data_to_acc(x_train, y_train, x_test, y_test, verbose=0, batch_size=8, lr=0.001):

  if len(y_train) == 0:
    return 0.1, None

  if x_train.shape[1]==28:
    x_train = np.moveaxis(x_train, 3, 1)
    x_test = np.moveaxis(x_test, 3, 1)

  if len(y_train.shape)>1 and y_train.shape[1]>1:
    y_train = np.argmax(y_train, axis=1)
  
  if len(y_test.shape)>1 and y_test.shape[1]>1:
    y_test = np.argmax(y_test, axis=1)

  y_train = y_train.reshape(-1)
  y_test = y_test.reshape(-1)

  criterion = torch.nn.CrossEntropyLoss()
  net = MnistLogistic().cuda()
  
  optimizer = torch.optim.Adam(net.parameters(), lr=lr)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)
  train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=True)
  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  max_acc = 0

  for epoch in range(30):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).long()
          optimizer.zero_grad()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
      correct = 0
      total = 0
      for images, labels in test_loader:
          images = Variable(images)
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()
      accuracy = correct/total
      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))

      max_acc = max(max_acc, accuracy.item())
  return max_acc, net


def torch_mnist_smallCNN_data_to_acc(x_train, y_train, x_test, y_test, verbose=0, batch_size=32):

  if len(y_train) == 0:
    return 0.1, None

  if x_train.shape[1]==28:
    x_train = np.moveaxis(x_train, 3, 1)
    x_test = np.moveaxis(x_test, 3, 1)

  if len(y_train.shape)>1 and y_train.shape[1]>1:
    y_train = np.argmax(y_train, axis=1)
  
  if len(y_test.shape)>1 and y_test.shape[1]>1:
    y_test = np.argmax(y_test, axis=1)

  y_train = y_train.reshape(-1)
  y_test = y_test.reshape(-1)

  criterion = torch.nn.CrossEntropyLoss()
  net = MnistLeNet().cuda()
  
  optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)
  train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=True)
  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  max_acc = 0

  for epoch in range(30):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).long()
          optimizer.zero_grad()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
      correct = 0
      total = 0
      for images, labels in test_loader:
          images = Variable(images)
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()
      accuracy = correct/total
      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))

      max_acc = max(max_acc, accuracy.item())

  return max_acc, net


def torch_mnist_largeCNN_data_to_acc(x_train, y_train, x_test, y_test, verbose=0, batch_size=32):

  if len(y_train) == 0:
    return 0.1, None

  if x_train.shape[1]==28:
    x_train = np.moveaxis(x_train, 3, 1)
    x_test = np.moveaxis(x_test, 3, 1)

  if len(y_train.shape)>1 and y_train.shape[1]>1:
    y_train = np.argmax(y_train, axis=1)
  
  if len(y_test.shape)>1 and y_test.shape[1]>1:
    y_test = np.argmax(y_test, axis=1)

  y_train = y_train.reshape(-1)
  y_test = y_test.reshape(-1)

  criterion = torch.nn.CrossEntropyLoss()
  net = MnistLargeCNN().cuda()
  
  optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), torch.Tensor(y_train).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)
  train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=True)
  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), torch.Tensor(y_test).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

  max_acc = 0

  for epoch in range(30):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).long()
          optimizer.zero_grad()
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
      correct = 0
      total = 0
      for images, labels in test_loader:
          images = Variable(images)
          logits = net(images)
          outputs = F.softmax(logits, dim=1)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()
      accuracy = correct/total
      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))

      max_acc = max(max_acc, accuracy.item())
  return max_acc, net




# utility_func_args = [x_train, y_train, x_val, y_val]
def sample_utility_and_cpt(n, size_min, size_max, utility_func, utility_func_args, random_state, save_dir, ub_prob=0, verbose=False):

  x_train, y_train, x_val, y_val = utility_func_args
  x_train, y_train = np.array(x_train), np.array(y_train)

  N = len(y_train)

  np.random.seed(random_state)
  
  for i in range(n):
    if verbose: print('Sample {} / {}'.format(i, n))

    n_select = np.random.choice(range(size_min, size_max))

    subset_index = []

    toss = np.random.uniform()

    # With probability ub_prob, sample a class-imbalanced subset
    if toss > 1-ub_prob:
      n_per_class = int(N / 10)
      alpha = np.ones(10)*30
      alpha[np.random.choice(range(10))] = np.random.choice(range(1, 50))
      p = np.random.dirichlet(alpha=alpha)
      occur = np.random.choice(range(10), size=n_select, replace=True, p=p)
      counts = np.array([np.sum(occur==i) for i in range(10)])
      for i in range(10):
        # ind_i = np.where(np.argmax(y_train, 1)==i)[0]
        ind_i = np.where(y_train==i)[0]
        if len(ind_i) > counts[i]:
          selected_ind_i = np.random.choice(ind_i, size=counts[i], replace=False)
        else:
          selected_ind_i = np.random.choice(ind_i, size=counts[i], replace=True)
        subset_index = subset_index + list(selected_ind_i)
      subset_index = np.array(subset_index)

    else:
      subset_index = np.random.choice(range(N), n_select, replace=False)

    subset_index = np.array(subset_index)
    acc, net = utility_func(x_train[subset_index], y_train[subset_index], x_val, y_val)

    PATH = save_dir + '_{}.cpt'.format(i)

    torch.save({'model_state_dict': net.state_dict(), 'subset_index': subset_index, 'accuracy': acc}, 
               PATH)
"""








