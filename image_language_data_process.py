from __future__ import absolute_import
import sys
import os
import os.path as osp
import errno
import network
from models import Resnext
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
from datasets import load_dataset
from transformers import ViTFeatureExtractor, ViTModel
from transformers import ViTImageProcessor
from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)
from transformers import ViTForImageClassification
import pickle
import shutil
import json
import numpy as np
from numpy import linalg as LA
import torch
import scipy

sys.path.append('.')
from sklearn.decomposition import PCA, KernelPCA
import math

SHAPES = {
    "cifar10": (32, 32, 3),
    "cifar10_500K": (32, 32, 3),
    "fmnist": (28, 28, 1),
    "mnist": (28, 28, 1),
    "svhn": (298, 28, 3)
}


def PrepareData(dataset, feature, num_query, dataset_path, seed):
    """
    Takes a dataset name and the size of the teacher ensemble and prepares
    training data for the student model, according to parameters indicated
    in flags above.
    :param dataset: string corresponding to mnist, cifar10, or svhn
    :param nb_teachers: number of teachers (in the ensemble) to learn from
    :param save: if set to True, will dump student training labels predicted by
                 the ensemble of teachers (with Laplacian noise) as npy files.
                 It also dumps the clean votes for each class (without noise) and
                 the labels assigned by teachers
    :return: pairs of (data, labels) to be used for student training and testing

    """
    # resnet50 requires a pre-process on the dataset loading
    if feature == 'resnet50':
        weight = ResNet50_Weights.IMAGENET1K_V2
        preprocess = weight.transforms()
        if dataset == 'cifar10':
            train_dataset = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), preprocess]
            ))
            test_dataset = datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), preprocess]
            ))
            test_labels = test_dataset.targets
            train_labels = train_dataset.targets
        elif dataset == 'fmnist':

            train_dataset = datasets.FashionMNIST(root=dataset_path, train=True, download=True,
                                                  transform=transforms.Compose(
                                                      [transforms.ToTensor(),
                                                       transforms.Lambda(lambda x: x.repeat(3, 1, 1)), preprocess]
                                                  ))
            test_dataset = datasets.FashionMNIST(root=dataset_path, train=False, download=True,
                                                 transform=transforms.Compose(
                                                     [transforms.ToTensor(),
                                                      transforms.Lambda(lambda x: x.repeat(3, 1, 1)), preprocess]
                                                 ))
            test_labels = test_dataset.targets
            train_labels = train_dataset.targets
        elif dataset == 'mnist':

            train_dataset = datasets.MNIST(root=dataset_path, train=True, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1)), preprocess]
            ))
            test_dataset = datasets.MNIST(root=dataset_path, train=False, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1)), preprocess]
            ))
            test_labels = test_dataset.targets
            train_labels = train_dataset.targets
        train_feature, test_feature = extract_feature(train_dataset, test_dataset, feature, dataset)

    # Load language datasets
    if dataset == 'sst2':
        ori_dataset = load_dataset('glue', 'sst2')
        train_dataset = ori_dataset['train']['sentence']
        test_dataset = ori_dataset['test']['sentence']
        train_labels = ori_dataset['train']['label']
        test_labels = ori_dataset['test']['label']
    elif dataset == 'agnews':
        ori_dataset = load_dataset('ag_news')
        train_dataset = ori_dataset['train']['text']
        test_dataset = ori_dataset['test']['text']
        train_labels = ori_dataset['train']['label']
        test_labels = ori_dataset['test']['label']
    elif dataset == 'dbpedia':
        ori_dataset = load_dataset('dbpedia_14')
        train_dataset = ori_dataset['train']['content']
        test_dataset = ori_dataset['test']['content']
        train_labels = ori_dataset['train']['label']
        test_labels = ori_dataset['test']['label']

    # Use vision transformer as the feature extractor
    if feature == 'vit':

        path = 'features/'
        train_path  = path + f'vit_{dataset}_train.npy'
        test_path = path + f'vit_{dataset}_test.npy'
        if os.path.exists(train_path):
            train_feature = np.load(train_path)
            test_feature = np.load(test_path)
        else:
            print(f'vit train feature is not found under path {train_path}')
            train_feature, test_feature = extract_feature(feature, dataset=dataset)

        # CIFAR-10 train labels obtained through load_dataset is different from that obtained from torchvision.datasets.
        train_ds, test_ds = load_dataset('cifar10', split=['train[:]', 'test[:]'])
        train_labels = train_ds['label']
        test_labels = test_ds['label']
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)
    elif feature == 'all-roberta-large-v1':

        train_path = f'features/{dataset}_{feature}_train.npy'
        test_path = f'features/{dataset}_{feature}_test.npy'
        if os.path.exists(train_path):
            train_feature = np.load(train_path)
            test_feature = np.load(test_path)
            # return train_feature, test_feature
        else:
            model = SentenceTransformer('all-roberta-large-v1')
            train_feature = model.encode(train_dataset)
            print('feature shape', train_feature.shape)
            test_feature = model.encode(test_dataset)
            np.save(f'features/{dataset}_{feature}_train.npy', train_feature)
            np.save(f'features/{dataset}_{feature}_test.npy', test_feature)

    train_mean = np.mean(train_feature, axis=0)
    train_var = np.var(train_feature, axis=0)
    test_mean = np.mean(test_feature, axis=0)
    test_var = np.var(test_feature, axis=0)
    train_feature_center = train_feature - train_mean
    test_feature_center = test_feature - test_mean
    train_l2_norm = LA.norm(train_feature_center, axis=1)
    test_l2_norm = LA.norm(test_feature_center, axis=1)
    train_feature_norm = train_feature_center / train_l2_norm[:, np.newaxis]
    test_feature_norm = test_feature_center / test_l2_norm[:, np.newaxis]
    print(f'test the first feature norm is {LA.norm(train_feature_norm[0, :])}')
    np.random.seed(seed)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    # repeat dataset for a fixed time
    random_index = np.random.randint(0, test_feature.shape[0], num_query).astype(int)
    return train_feature_norm, train_labels, test_feature_norm[random_index], test_labels[random_index]


def extract_label(dataset, name):
    if os.path.exists(f'{name}_label.npy'):
        label_list = np.load(f'{name}_label.npy')
        print('label_shape', label_list.shape)
    else:
        label_list = []
        for idx, (imgs, label) in enumerate(dataset):
            label_list.append(label)
        label_list = np.array(label_list)
        np.save(f'{name}_label.npy', label_list)

    return label_list


def extract_feature(feature, train_datapoint=None, test_datapoint=None, dataset='cifar10', feature_path='features/'):
    """
    Extract features with the pre-trained Resnet-50 model, visition transformer, and the sentence transformer.
    :param FLAGS:
    :param ckpt_path:
    :return:
    """

    if feature == 'resnet50':

        weight = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weight)
        model.eval()
        print('len of data', len(train_datapoint))
        train_path = f'{feature_path}{dataset}_{feature}_train.npy'
        test_path = f'{feature_path}{dataset}_{feature}_test.npy'
        if os.path.exists(train_path):
            train_feature = np.load(train_path)
        else:
            print('file does not exist')
            train_feature = network.predFeature(model, train_datapoint)
            np.save(train_path, train_feature)
        if os.path.exists(test_path):
            test_feature = np.load(test_path)
        else:
            test_feature = network.predFeature(model, test_datapoint)
            np.save(test_path, test_feature)
        print(f'feature shape is {train_feature.shape}')

        return train_feature, test_feature

    elif feature == 'vit':

        train_ds, test_ds = load_dataset(dataset, split=['train[:]', 'test[:]'])
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        model.eval()
        image_mean, image_std = processor.image_mean, processor.image_std
        size = processor.size["height"]
        normalize = Normalize(mean=image_mean, std=image_std)

        _val_transforms = Compose(
            [
                Resize(size),
                ToTensor(),
                normalize,
            ]
        )

        def val_transforms(examples):
            if dataset == 'mnist':
                examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
            else:
                examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]

            return examples

        # Set the transforms
        train_ds.set_transform(val_transforms)
        # val_ds.set_transform(val_transforms)
        test_ds.set_transform(val_transforms)

        def collate_fn(examples):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            labels = torch.tensor([example["label"] for example in examples])
            return {"pixel_values": pixel_values, "labels": labels}

        use_gpu = torch.cuda.is_available()

        if use_gpu:
            # print("Currently using GPU {}".format(config.gpu_devices))
            cudnn.benchmark = True
            torch.cuda.manual_seed_all(0)
            trainloader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=200)
            testloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=200)
        else:
            print("Currently using CPU (GPU is highly recommended)")
            pin_memory = True if use_gpu else False
            trainloader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=200)
            testloader = DataLoader(test_ds, collate_fn, batch_size=200)

        if use_gpu:
            model = nn.DataParallel(model).cuda()
        model.eval()
        os.makedirs(feature_path, exist_ok=True)

        def extract_feature_vit(loader):
            with torch.no_grad():
                pred_list, feature_list = [], []
                for batch_idx, batch in enumerate(loader):
                    img_tuple, label = batch.items()
                    img = img_tuple[1]
                    # print('imgs,', img, 'label', label)
                    if batch_idx == 0:
                        print('image before pretrain', img.shape)
                    if batch_idx % 50 == 0:
                        print('batch {}/{}', batch_idx, len(loader))
                    output = model(img)
                    features = output.last_hidden_state[:, 0, :]
                    feature_list.append(features.cpu())
            feature_list = (((torch.cat(feature_list, 0)).float()).numpy()).tolist()
            feature_list = np.array(feature_list)
            return feature_list

        features_train = extract_feature_vit(trainloader)
        features_test = extract_feature_vit(testloader)
        np.save(f"{feature_path}vit_cifar10_train.npy", features_train)
        np.save(f"{feature_path}vit_cifar10_test.npy", features_test)
        print('size of train', features_train.shape, 'size of test', features_test.shape)
        return features_train, features_test



    elif feature == 'all-roberta-large-v1':

        train_path = f'features/{dataset}_{feature}_train.npy'
        test_path = f'features/{dataset}_{feature}_test.npy'
        if os.path.exists(train_path):
            train_feature = np.load(train_path)
            test_feature = np.load(test_path)

        else:
            model = SentenceTransformer('all-roberta-large-v1')
            train_feature = model.encode(train_datapoint)
            print('feature shape', train_feature.shape)
            test_feature = model.encode(test_datapoint)
            np.save(f'features/{dataset}_{feature}_train.npy', train_feature)
            np.save(f'features/{dataset}_{feature}_test.npy', test_feature)

        return train_feature, test_feature
