import os
import json
import logging

import numpy as np
import torch

from src.argument import parser, print_args

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

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_args(args, file_name):
    with open(file_name, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def load_args(file_name):
    args = parser()
    with open(file_name, 'r') as f:
        args.__dict__ = json.load(f)
    return args