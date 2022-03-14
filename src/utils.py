import os
import json
import logging

import numpy as np
import torch
import IPython

from src.argument import create_parser, print_args

def create_logger(save_path='', file_type='', level='debug'):

    if level == 'debug':
        _level = logging.DEBUG
    elif level == 'info':
        _level = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(_level)

    cs = logging.StreamHandler()
    cs.setLevel(_level)
    logger.addHandler(cs)

    if save_path != '':
        file_name = os.path.join(save_path, file_type + '_log.txt')
        fh = logging.FileHandler(file_name, mode='w')
        fh.setLevel(_level)

        logger.addHandler(fh)

    return logger


def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)


def load_model(model, file_name):
    model.load_state_dict(
            torch.load(file_name, map_location=lambda storage, loc: storage))

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_args(args, file_name):
    with open(file_name, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def load_args(file_name):
    parser = create_parser()
    args = parser.parse_args()
    # args = parser() # TODO
    with open(file_name, 'r') as f:
        args.__dict__ = json.load(f)
    return args


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=3, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, test_loss):
        if self.best_loss == None:
            self.best_loss = test_loss
        elif self.best_loss - test_loss > self.min_delta:
            self.best_loss = test_loss
        elif self.best_loss - test_loss < self.min_delta:
            self.counter += 1
            # print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


class EarlyStoppingBatch():
    """
    Like EarlyStopping, but stops when all members of batch meet the individual stopping criteria.
    Note: this is used for attacks, so loss is being maximized
    """
    def __init__(self, bs, patience=3, min_delta=1e-1):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta

        self.counter = torch.zeros(bs)
        self.best_loss = None #torch.ones(bs)*float("inf")
        self.early_stop_vec = torch.zeros(bs, dtype=torch.bool)
        self.early_stop = False

    def __call__(self, test_loss):
        if self.best_loss == None:
            self.best_loss = test_loss

        # print("Inside EarlyStoppingBatch")
        # IPython.embed()

        # print(self.best_loss, test_loss)
        improve_ind = torch.nonzero(test_loss - self.best_loss >= self.min_delta)
        nonimprove_ind = torch.nonzero(test_loss - self.best_loss < self.min_delta)
        self.best_loss[improve_ind] = test_loss[improve_ind]

        self.counter[nonimprove_ind] = self.counter[nonimprove_ind] + 1

        early_stop_ind = torch.nonzero(self.counter >= self.patience)
        self.early_stop_vec[early_stop_ind] = True

        # print(self.counter)
        if torch.all(self.early_stop_vec).item():
            print('INFO: Early stopping')
            self.early_stop = True

