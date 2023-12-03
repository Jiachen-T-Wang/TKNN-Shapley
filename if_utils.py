# PyTorch Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import grad

from torch.utils.data import Dataset, DataLoader, TensorDataset

# General
import pandas as pd 
import copy
import numpy as np
import pickle

import time
import datetime
import logging
import sys

from pathlib import Path

from utility_func import *
from helper import *
from models import *
import config

OpenML_dataset = config.OpenML_dataset



def get_default_config():
    """Returns a default config file"""
    config = {
        'outdir': 'outdir',
        'seed': 42,
        'gpu': 0,
        'dataset': 'CIFAR10',
        'num_classes': 10,
        'test_sample_num': 1,
        'test_start_index': 0,
        'recursion_depth': 1,
        'r_averaging': 1,
        'scale': None,
        'damp': None,
        'calc_method': 'img_wise',
        'log_filename': None,
    }

    return config


def display_progress(text, current_step, last_step, enabled=True,
                     fix_zero_start=True):
    """Draws a progress indicator on the screen with the text preceeding the
    progress

    Arguments:
        test: str, text displayed to describe the task being executed
        current_step: int, current step of the iteration
        last_step: int, last possible step of the iteration
        enabled: bool, if false this function will not execute. This is
            for running silently without stdout output.
        fix_zero_start: bool, if true adds 1 to each current step so that the
            display starts at 1 instead of 0, which it would for most loops
            otherwise.
    """
    if not enabled:
        return

    # Fix display for most loops which start with 0, otherwise looks weird
    if fix_zero_start:
        current_step = current_step + 1

    term_line_len = 80
    final_chars = [':', ';', ' ', '.', ',']
    if text[-1:] not in final_chars:
        text = text + ' '
    if len(text) < term_line_len:
        bar_len = term_line_len - (len(text)
                                   + len(str(current_step))
                                   + len(str(last_step))
                                   + len("  / "))
    else:
        bar_len = 30
    filled_len = int(round(bar_len * current_step / float(last_step)))
    bar = '=' * filled_len + '.' * (bar_len - filled_len)

    bar = f"{text}[{bar:s}] {current_step:d} / {last_step:d}"
    if current_step < last_step-1:
        # Erase to end of line and print
        sys.stdout.write("\033[K" + bar + "\r")
    else:
        sys.stdout.write(bar + "\n")

    sys.stdout.flush()



def s_test(z_test, t_test, model, z_loader, gpu=-1, damp=0.01, scale=25.0,
           recursion_depth=5000):
    """s_test can be precomputed for each test point of interest, and then
    multiplied with grad_z to get the desired value for each training point.
    Here, strochastic estimation is used to calculate s_test. s_test is the
    Inverse Hessian Vector Product.

    Arguments:
        z_test: torch tensor, test data points, such as test images
        t_test: torch tensor, contains all test data labels
        model: torch NN, model used to evaluate the dataset
        z_loader: torch Dataloader, can load the training dataset
        gpu: int, GPU id to use if >=0 and -1 means use CPU
        damp: float, dampening factor
        scale: float, scaling factor
        recursion_depth: int, number of iterations aka recursion depth
            should be enough so that the value stabilises.

    Returns:
        h_estimate: list of torch tensors, s_test"""
    v = grad_z(z_test, t_test, model, gpu)
    h_estimate = v.copy()

    ################################
    # TODO: Dynamically set the recursion depth so that iterations stops
    # once h_estimate stabilises
    ################################
    for i in range(recursion_depth):
        # take just one random sample from training dataset
        # easiest way to just use the DataLoader once, break at the end of loop
        #########################
        # TODO: do x, t really have to be chosen RANDOMLY from the train set?
        #########################
        for x, t in z_loader:
            if gpu >= 0:
                x, t = x.cuda(), t.cuda()
            y = model(x)
            loss = calc_loss(y, t)
            params = [ p for p in model.parameters() if p.requires_grad ]
            hv = hvp(loss, params, h_estimate)
            # Recursively caclulate h_estimate
            h_estimate = [
                _v + (1 - damp) * _h_e - _hv / scale
                for _v, _h_e, _hv in zip(v, h_estimate, hv)]
            break
    return h_estimate


def calc_loss(y, t):
    """Calculates the loss

    Arguments:
        y: torch tensor, input with size (minibatch, nr_of_classes)
        t: torch tensor, target expected by loss of size (0 to nr_of_classes-1)

    Returns:
        loss: scalar, the loss"""
    ####################
    # if dim == [0, 1, 3] then dim=0; else dim=1
    ####################
    # y = torch.nn.functional.log_softmax(y, dim=0)
    
    ### remove comment for CIFAR
    y = torch.nn.functional.log_softmax(y)
    loss = torch.nn.functional.nll_loss(y, t, weight=None, reduction='mean')

    ### Only for DOG VS CAT
    # loss = torch.nn.functional.binary_cross_entropy_with_logits(y, t.unsqueeze(1), weight=None, reduction='mean')

    return loss


def grad_z(z, t, model, gpu=-1):
    """Calculates the gradient z. One grad_z should be computed for each
    training sample.

    Arguments:
        z: torch tensor, training data points
            e.g. an image sample (batch_size, 3, 256, 256)
        t: torch tensor, training data labels
        model: torch NN, model used to evaluate the dataset
        gpu: int, device id to use for GPU, -1 for CPU

    Returns:
        grad_z: list of torch tensor, containing the gradients
            from model parameters to loss"""
    model.eval()
    # initialize
    if gpu >= 0:
        z, t = z.cuda(), t.cuda()
    y = model(z)
    loss = calc_loss(y, t)
    # Compute sum of gradients from model parameters to loss
    params = [ p for p in model.parameters() if p.requires_grad ]
    return list(grad(loss, params, create_graph=True))


def hvp(y, w, v):
    """Multiply the Hessians of y and w by v.
    Uses a backprop-like approach to compute the product between the Hessian
    and another vector efficiently, which even works for large Hessians.
    Example: if: y = 0.5 * w^T A x then hvp(y, w, v) returns and expression
    which evaluates to the same values as (A + A.t) v.

    Arguments:
        y: scalar/tensor, for example the output of the loss function
        w: list of torch tensors, tensors over which the Hessian
            should be constructed
        v: list of torch tensors, same shape as w,
            will be multiplied with the Hessian

    Returns:
        return_grads: list of torch tensors, contains product of Hessian and v.

    Raises:
        ValueError: `y` and `w` have a different length."""
    if len(w) != len(v):
        raise(ValueError("w and v must have the same length."))

    # First backprop
    first_grads = grad(y, w, retain_graph=True, create_graph=True)

    # Elementwise products
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)

    # Second backprop
    return_grads = grad(elemwise_products, w, create_graph=True)

    return return_grads





def calc_s_test(model, test_loader, train_loader, save=False, gpu=-1,
                damp=0.01, scale=25, recursion_depth=5000, r=1, start=0):
    """Calculates s_test for the whole test dataset taking into account all
    training data images.

    Arguments:
        model: pytorch model, for which s_test should be calculated
        test_loader: pytorch dataloader, which can load the test data
        train_loader: pytorch dataloader, which can load the train data
        save: Path, path where to save the s_test files if desired. Omitting
            this argument will skip saving
        gpu: int, device id to use for GPU, -1 for CPU (default)
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
        start: int, index of the first test index to use. default is 0

    Returns:
        s_tests: list of torch vectors, contain all s_test for the whole
            dataset. Can be huge.
        save: Path, path to the folder where the s_test files were saved to or
            False if they were not saved."""
    if save and not isinstance(save, Path):
        save = Path(save)
    if not save:
        logging.info("ATTENTION: not saving s_test files.")

    s_tests = []
    for i in range(start, len(test_loader.dataset)):
        z_test, t_test = test_loader.dataset[i]
        z_test = test_loader.collate_fn([z_test])
        t_test = test_loader.collate_fn([t_test])

        s_test_vec = calc_s_test_single(model, z_test, t_test, train_loader,
                                        gpu, damp, scale, recursion_depth, r)

        if save:
            s_test_vec = [s.cpu() for s in s_test_vec]
            torch.save(
                s_test_vec,
                save.joinpath(f"{i}_recdep{recursion_depth}_r{r}.s_test"))
        else:
            s_tests.append(s_test_vec)
        display_progress(
            "Calc. z_test (s_test): ", i-start, len(test_loader.dataset)-start)

    return s_tests, save


def calc_s_test_single(model, z_test, t_test, train_loader, gpu=-1,
                       damp=0.01, scale=25, recursion_depth=5000, r=1):
    """Calculates s_test for a single test image taking into account the whole
    training dataset. s_test = invHessian * nabla(Loss(test_img, model params))

    Arguments:
        model: pytorch model, for which s_test should be calculated
        z_test: test image
        t_test: test image label
        train_loader: pytorch dataloader, which can load the train data
        gpu: int, device id to use for GPU, -1 for CPU (default)
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.

    Returns:
        s_test_vec: torch tensor, contains s_test for a single test image"""
    s_test_vec_list = []
    for i in range(r):
        s_test_vec_list.append(s_test(z_test, t_test, model, train_loader,
                                      gpu=gpu, damp=damp, scale=scale,
                                      recursion_depth=recursion_depth))

    ################################
    # TODO: Understand why the first[0] tensor is the largest with 1675 tensor
    #       entries while all subsequent ones only have 335 entries?
    ################################
    s_test_vec = s_test_vec_list[0]
    for i in range(1, r):
        s_test_vec += s_test_vec_list[i]

    s_test_vec = [i / r for i in s_test_vec]

    return s_test_vec


def calc_grad_z(model, train_loader, save_pth=False, gpu=-1, start=0):
    """Calculates grad_z and can save the output to files. One grad_z should
    be computed for each training data sample.

    Arguments:
        model: pytorch model, for which s_test should be calculated
        train_loader: pytorch dataloader, which can load the train data
        save_pth: Path, path where to save the grad_z files if desired.
            Omitting this argument will skip saving
        gpu: int, device id to use for GPU, -1 for CPU (default)
        start: int, index of the first test index to use. default is 0

    Returns:
        grad_zs: list of torch tensors, contains the grad_z tensors
        save_pth: Path, path where grad_z files were saved to or
            False if they were not saved."""
    if save_pth and isinstance(save_pth, str):
        save_pth = Path(save_pth)
    if not save_pth:
        logging.info("ATTENTION: Not saving grad_z files!")

    grad_zs = []
    for i in range(start, len(train_loader.dataset)):
        z, t = train_loader.dataset[i]
        z = train_loader.collate_fn([z])
        t = train_loader.collate_fn([t])
        grad_z_vec = grad_z(z, t, model, gpu=gpu)
        if save_pth:
            grad_z_vec = [g.cpu() for g in grad_z_vec]
            torch.save(grad_z_vec, save_pth.joinpath(f"{i}.grad_z"))
        else:
            grad_zs.append(grad_z_vec)

    return grad_zs, save_pth


def load_s_test(s_test_dir=Path("./s_test/"), s_test_id=0, r_sample_size=10,
                train_dataset_size=-1):
    """Loads all s_test data required to calculate the influence function
    and returns a list of it.

    Arguments:
        s_test_dir: Path, folder containing files storing the s_test values
        s_test_id: int, number of the test data sample s_test was calculated
            for
        r_sample_size: int, number of s_tests precalculated
            per test dataset point
        train_dataset_size: int, number of total samples in dataset;
            -1 indicates to use all available grad_z files

    Returns:
        e_s_test: list of torch vectors, contains all e_s_tests for the whole
            dataset.
        s_test: list of torch vectors, contain all s_test for the whole
            dataset. Can be huge."""

    if isinstance(s_test_dir, str):
        s_test_dir = Path(s_test_dir)

    s_test = []
    logging.info(f"Loading s_test from: {s_test_dir} ...")
    num_s_test_files = len(s_test_dir.glob("*.s_test"))
    if num_s_test_files != r_sample_size:
        logging.warn("Load Influence Data: number of s_test sample files"
                     " mismatches the available samples")
    ########################
    # TODO: should prob. not hardcode the file name, use natsort+glob
    ########################
    for i in range(num_s_test_files):
        s_test.append(
            torch.load(s_test_dir / str(s_test_id) + f"_{i}.s_test"))
        display_progress("s_test files loaded: ", i, r_sample_size)

    #########################
    # TODO: figure out/change why here element 0 is chosen by default
    #########################
    e_s_test = s_test[0]
    # Calculate the sum
    for i in range(len(s_test)):
        e_s_test = [i + j for i, j in zip(e_s_test, s_test[0])]

    # Calculate the average
    #########################
    # TODO: figure out over what to calculate the average
    #       should either be r_sample_size OR e_s_test
    #########################
    e_s_test = [i / len(s_test) for i in e_s_test]

    return e_s_test, s_test


def load_grad_z(grad_z_dir=Path("./grad_z/"), train_dataset_size=-1):
    """Loads all grad_z data required to calculate the influence function and
    returns it.

    Arguments:
        grad_z_dir: Path, folder containing files storing the grad_z values
        train_dataset_size: int, number of total samples in dataset;
            -1 indicates to use all available grad_z files

    Returns:
        grad_z_vecs: list of torch tensors, contains the grad_z tensors"""
    if isinstance(grad_z_dir, str):
        grad_z_dir = Path(grad_z_dir)

    grad_z_vecs = []
    logging.info(f"Loading grad_z from: {grad_z_dir} ...")
    available_grad_z_files = len(grad_z_dir.glob("*.grad_z"))
    if available_grad_z_files != train_dataset_size:
        logging.warn("Load Influence Data: number of grad_z files mismatches"
                     " the dataset size")
        if -1 == train_dataset_size:
            train_dataset_size = available_grad_z_files
    for i in range(train_dataset_size):
        grad_z_vecs.append(torch.load(grad_z_dir / str(i) + ".grad_z"))
        display_progress("grad_z files loaded: ", i, train_dataset_size)

    return grad_z_vecs


def calc_influence_function(train_dataset_size, grad_z_vecs=None,
                            e_s_test=None):
    """Calculates the influence function

    Arguments:
        train_dataset_size: int, total train dataset size
        grad_z_vecs: list of torch tensor, containing the gradients
            from model parameters to loss
        e_s_test: list of torch tensor, contains s_test vectors

    Returns:
        influence: list of float, influences of all training data samples
            for one test sample
        harmful: list of float, influences sorted by harmfulness
        helpful: list of float, influences sorted by helpfulness"""
    if not grad_z_vecs and not e_s_test:
        grad_z_vecs = load_grad_z()
        e_s_test, _ = load_s_test(train_dataset_size=train_dataset_size)

    if (len(grad_z_vecs) != train_dataset_size):
        logging.warn("Training data size and the number of grad_z files are"
                     " inconsistent.")
        train_dataset_size = len(grad_z_vecs)

    influences = []
    for i in range(train_dataset_size):
        tmp_influence = -sum(
            [
                ###################################
                # TODO: verify if computation really needs to be done
                # on the CPU or if GPU would work, too
                ###################################
                torch.sum(k * j).data.cpu().numpy()
                for k, j in zip(grad_z_vecs[i], e_s_test)
                ###################################
                # Originally with [i] because each grad_z contained
                # a list of tensors as long as e_s_test list
                # There is one grad_z per training data sample
                ###################################
            ]) / train_dataset_size
        influences.append(tmp_influence)
        display_progress("Calc. influence function: ", i, train_dataset_size)

    harmful = np.argsort(influences)
    helpful = harmful[::-1]

    return influences, harmful.tolist(), helpful.tolist()


def calc_influence_single(model, train_loader, test_loader, test_id_num, gpu,
                          recursion_depth, r, s_test_vec=None,
                          time_logging=False):
    """Calculates the influences of all training data points on a single
    test dataset image.

    Arugments:
        model: pytorch model
        train_loader: DataLoader, loads the training dataset
        test_loader: DataLoader, loads the test dataset
        test_id_num: int, id of the test sample for which to calculate the
            influence function
        gpu: int, identifies the gpu id, -1 for cpu
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
        s_test_vec: list of torch tensor, contains s_test vectors. If left
            empty it will also be calculated

    Returns:
        influence: list of float, influences of all training data samples
            for one test sample
        harmful: list of float, influences sorted by harmfulness
        helpful: list of float, influences sorted by helpfulness
        test_id_num: int, the number of the test dataset point
            the influence was calculated for"""
    # Calculate s_test vectors if not provided
    if not s_test_vec:
        z_test, t_test = test_loader.dataset[test_id_num]
        z_test = test_loader.collate_fn([z_test])
        t_test = test_loader.collate_fn([t_test])
        s_test_vec = calc_s_test_single(model, z_test, t_test, train_loader,
                                        gpu, recursion_depth=recursion_depth,
                                        r=r)

    # Calculate the influence function
    train_dataset_size = len(train_loader.dataset)
    influences = []
    for i in range(train_dataset_size):
        z, t = train_loader.dataset[i]
        z = train_loader.collate_fn([z])
        t = train_loader.collate_fn([t])
        if time_logging:
            time_a = datetime.datetime.now()
        grad_z_vec = grad_z(z, t, model, gpu=gpu)
        if time_logging:
            time_b = datetime.datetime.now()
            time_delta = time_b - time_a
            logging.info(f"Time for grad_z iter:"
                         f" {time_delta.total_seconds() * 1000}")
        tmp_influence = -sum(
            [
                ####################
                # TODO: potential bottle neck, takes 17% execution time
                # torch.sum(k * j).data.cpu().numpy()
                ####################
                torch.sum(k * j).data
                for k, j in zip(grad_z_vec, s_test_vec)
            ]) / train_dataset_size
        influences.append(tmp_influence)
        # display_progress("Calc. influence function: ", i, train_dataset_size)

    return influences, test_id_num


def get_dataset_sample_ids_per_class(class_id, num_samples, test_loader,
                                     start_index=0):
    """Gets the first num_samples from class class_id starting from
    start_index. Returns a list with the indicies which can be passed to
    test_loader.dataset[X] to retreive the actual data.

    Arguments:
        class_id: int, name or id of the class label
        num_samples: int, number of samples per class to process
        test_loader: DataLoader, can load the test dataset.
        start_index: int, means after which x occourance to add an index
            to the list of indicies. E.g. if =3, then it would add the
            4th occourance of an item with the label class_nr to the list.

    Returns:
        sample_list: list of int, contains indicies of the relevant samples"""
    #######
    # NOTE / TODO: here's optimisation potential. We are currently searching
    # for the x+1th sample and when that's found we cancel the loop. we could
    # stop after finding the x'th picture (start_index + num_samples)
    #######
    sample_list = []
    img_count = 0
    for i in range(len(test_loader.dataset)):
        _, t = test_loader.dataset[i]
        if class_id == t:
            img_count += 1
            if (img_count > start_index) and \
                    (img_count <= start_index + num_samples):
                sample_list.append(i)
            elif img_count > start_index + num_samples:
                break

    return sample_list


def get_dataset_sample_ids(num_samples, test_loader, num_classes=None,
                           start_index=0):
    """Gets the first num_sample indices of all classes starting from
    start_index per class. Returns a list and a dict containing the indicies.

    Arguments:
        num_samples: int, number of samples of each class to return
        test_loader: DataLoader, can load the test dataset
        num_classes: int, number of classes contained in the dataset
        start_index: int, means after which x occourance to add an index
            to the list of indicies. E.g. if =3, then it would add the
            4th occourance of an item with the label class_nr to the list.

    Returns:
        sample_dict: dict, containing dict[class] = list_of_indices
        sample_list: list, containing a continious list of indices"""
    sample_dict = {}
    sample_list = []
    if not num_classes:
        num_classes = len(np.unique(test_loader.dataset.targets))
    if start_index > len(test_loader.dataset) / num_classes:
        logging.warn(f"The variable test_start_index={start_index} is "
                     f"larger than the number of available samples per class.")
    for i in range(num_classes):
        sample_dict[str(i)] = get_dataset_sample_ids_per_class(
            i, num_samples, test_loader, start_index)
        # Append the new list on the same level as the old list
        # Avoids having a list of lists
        sample_list[len(sample_list):len(sample_list)] = sample_dict[str(i)]
    return sample_dict, sample_list


def calc_img_wise(config, model, train_loader, test_loader):
    """Calculates the influence function one test point at a time. Calculates
    the `s_test` and `grad_z` values on the fly and discards them afterwards.

    Arguments:
        config: dict, contains the configuration from cli params"""

    test_dataset_iter_len = len(test_loader.dataset)
    influences = {}

    # Main loop for calculating the influence function one test sample per iteration.
    for j in range(test_dataset_iter_len):
        
        i = j

        start_time = time.time()
        influence, _ = calc_influence_single(
            model, train_loader, test_loader, test_id_num=i, gpu=config['gpu'],
            recursion_depth=config['recursion_depth'], r=config['r_averaging'])
        end_time = time.time()

        influences[str(i)] = {}
        _, label = test_loader.dataset[i]
        influences[str(i)]['label'] = label
        influences[str(i)]['num_in_dataset'] = j
        influences[str(i)]['time_calc_influence_s'] = end_time - start_time
        infl = [x.cpu().numpy().tolist() for x in influence]
        influences[str(i)]['influence'] = infl

    return influences


def calc_all_grad_then_test(config, model, train_loader, test_loader):
    """Calculates the influence function by first calculating
    all grad_z, all s_test and then loading them to calc the influence"""

    outdir = Path(config['outdir'])
    outdir.mkdir(exist_ok=True, parents=True)

    s_test_outdir = outdir.joinpath("s_test/")
    if not s_test_outdir.exists():
        s_test_outdir.mkdir()
    grad_z_outdir = outdir.joinpath("grad_z/")
    if not grad_z_outdir.exists():
        grad_z_outdir.mkdir()

    influence_results = {}

    calc_s_test(model, test_loader, train_loader, s_test_outdir,
                config['gpu'], config['damp'], config['scale'],
                config['recursion_depth'], config['r_averaging'],
                config['test_start_index'])
    calc_grad_z(model, train_loader, grad_z_outdir, config['gpu'],
                config['test_start_index'])

    train_dataset_len = len(train_loader.dataset)
    influences, _, _ = calc_influence_function(train_dataset_len)

    influence_results['influences'] = influences



def total_influence(influences, n_train, n_test):
  tot = np.zeros(n_train)
  for i in range(n_test):
    tot += influences[str(i)]['influence']
  return tot / n_test


def cifar_smallCNN_influence_score(x_train, y_train, x_test, y_test, verbose=0):

  if x_train.shape[1]==32:
    x_train = np.moveaxis(x_train, 3, 1)
    x_test = np.moveaxis(x_test, 3, 1)

  y_train = y_train.reshape(-1)
  y_test = y_test.reshape(-1)

  criterion = torch.nn.CrossEntropyLoss()
  net = SmallCNN_CIFAR().cuda()
  optimizer = torch.optim.Adam(net.parameters(), lr=0.001, eps=1e-7)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), (torch.Tensor(y_train)).long().cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)
  train_loader = DataLoader(dataset=fewshot_dataset, batch_size=32, shuffle=True)
  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), (torch.Tensor(y_test)).long().cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

  for epoch in range(10):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels)
          optimizer.zero_grad()
          outputs = net(images)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
      correct = 0
      total = 0
      for images, labels in test_loader:
          images = Variable(images)
          outputs = net(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()
      accuracy = 100 * correct/total
      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))

  config = get_default_config()
  influences = calc_img_wise(config, net.cuda(), train_loader, test_loader)
  total_inf = total_influence(influences, n_train=len(train_loader.dataset), n_test=len(test_loader.dataset))
  return -total_inf



def cifar_resnet_influence_score(x_train, y_train, x_test, y_test, verbose=0):

  if x_train.shape[1]==32:
    x_train = np.moveaxis(x_train, 3, 1)
    x_test = np.moveaxis(x_test, 3, 1)

  y_train = y_train.reshape(-1)
  y_test = y_test.reshape(-1)

  criterion = torch.nn.CrossEntropyLoss()
  net = ResNet18().cuda()
  optimizer = torch.optim.Adam(net.parameters(), lr=0.001, eps=1e-7)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), (torch.Tensor(y_train)).long().cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)
  train_loader = DataLoader(dataset=fewshot_dataset, batch_size=32, shuffle=True)
  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), (torch.Tensor(y_test)).long().cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

  for epoch in range(10):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels)
          optimizer.zero_grad()
          outputs = net(images)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
      correct = 0
      total = 0
      for images, labels in test_loader:
          images = Variable(images)
          outputs = net(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum()
      accuracy = 100 * correct/total
      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))

  config = get_default_config()
  influences = calc_img_wise(config, net.cuda(), train_loader, test_loader)
  total_inf = total_influence(influences, n_train=len(train_loader.dataset), n_test=len(test_loader.dataset))
  return -total_inf




def pubfig_smallCNN_influence_score(x_train, y_train, x_test, y_test, verbose=0):

  if x_train.shape[1]==32:
    x_train = np.moveaxis(x_train, 3, 1)
    x_test = np.moveaxis(x_test, 3, 1)

  y_train = y_train.reshape(-1)
  y_test = y_test.reshape(-1)

  criterion = torch.nn.CrossEntropyLoss()
  net = SmallCNN_PUBFIG().cuda()
  optimizer = torch.optim.Adam(net.parameters(), lr=0.001, eps=1e-7)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), (torch.Tensor(y_train)).long().cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)
  train_loader = DataLoader(dataset=fewshot_dataset, batch_size=32, shuffle=True)
  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), (torch.Tensor(y_test)).long().cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

  for epoch in range(30):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels)
          optimizer.zero_grad()
          outputs = net(images)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

  config = get_default_config()
  influences = calc_img_wise(config, net.cuda(), train_loader, test_loader)
  total_inf = total_influence(influences, n_train=len(train_loader.dataset), n_test=len(test_loader.dataset))
  return -total_inf





def dogcat_smallCNN_influence_score(x_train, y_train, x_test, y_test, verbose=0):

  if x_train.shape[1]==32:
    x_train = np.moveaxis(x_train, 3, 1)
    x_test = np.moveaxis(x_test, 3, 1)

  y_train = y_train.reshape(-1)
  y_test = y_test.reshape(-1)

  criterion = torch.nn.BCEWithLogitsLoss()
  net = SmallCNN_DOGCAT().cuda()
  optimizer = torch.optim.Adam(net.parameters(), lr=0.001, eps=1e-7)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), (torch.Tensor(y_train)).cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)
  train_loader = DataLoader(dataset=fewshot_dataset, batch_size=32, shuffle=True)
  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), (torch.Tensor(y_test)).cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

  for epoch in range(10):
      for i, (images, labels) in enumerate(train_loader):
          images = Variable(images)
          labels = Variable(labels).float()
          optimizer.zero_grad()
          outputs = net(images)
          loss = criterion(outputs, labels.unsqueeze(1))
          loss.backward()
          optimizer.step()
      correct = 0
      total = 0
      for images, labels in test_loader:
          images = Variable(images)
          outputs = net(images)
          # _, predicted = torch.max(outputs.data, 1)
          predicted = torch.round( torch.sigmoid(outputs.data) )
          total += labels.size(0)
          correct += (predicted == labels.unsqueeze(1)).sum()
      accuracy = 100 * correct/total
      if verbose:
        print("Epoch: {}. Loss: {}. Accuracy: {}.".format(epoch, loss.item(), accuracy))

  config = get_default_config()
  influences = calc_img_wise(config, net.cuda(), train_loader, test_loader)
  total_inf = total_influence(influences, n_train=len(train_loader.dataset), n_test=len(test_loader.dataset))
  return -total_inf



def compute_influence_score(dataset, model_type, x_train, y_train, x_test, y_test, batch_size=32, lr=0.001, verbose=0):

  # Process Datasets and Initialize Models
  if dataset in ['MNIST', 'FMNIST']:

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

  elif dataset in OpenML_dataset:

    if model_type == 'MLP':
        net = BinaryMLP( x_train.shape[1], max(100, int(x_train.shape[1]/2)) ).cuda()
    elif model_type == 'MLP_RS':
        net = BinaryMLP( x_train.shape[1], max(100, int(x_train.shape[1]/2)) ).cuda()
    else:
        print('not supported')
    
    criterion = torch.nn.CrossEntropyLoss()
    n_epoch = 30
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-7)

  else:
    print('dataset not supported')
    sys.exit(1)

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), (torch.Tensor(y_train)).long().cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)
  train_loader = DataLoader(dataset=fewshot_dataset, batch_size=32, shuffle=True)
  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), (torch.Tensor(y_test)).long().cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

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

  tensor_x, tensor_y = torch.Tensor(x_train).cuda(), (torch.Tensor(y_train)).long().cuda()
  fewshot_dataset = TensorDataset(tensor_x,tensor_y)
  train_loader = DataLoader(dataset=fewshot_dataset, batch_size=32, shuffle=True)
  tensor_x_test, tensor_y_test = torch.Tensor(x_test).cuda(), (torch.Tensor(y_test)).long().cuda()
  test_dataset = TensorDataset(tensor_x_test,tensor_y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

  config = get_default_config()
  influences = calc_img_wise(config, net.cuda(), train_loader, test_loader)
  total_inf = total_influence(influences, n_train=len(train_loader.dataset), n_test=len(test_loader.dataset))

  return -total_inf




