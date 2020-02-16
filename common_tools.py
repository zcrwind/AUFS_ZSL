# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.init as init

def print_args(args):
    print('-' * 50)
    for arg, content in args.__dict__.items():
        print("{}: {}".format(arg, content))
    print('-' * 50)


def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)


def reset_grad(nets):
    for net in nets:
        net.zero_grad()


def nets_weights_init(nets):
    for net in nets:
        net.apply(weights_init)


def print_nets(nets):
    for net in nets:
        print(net)


def cosine_distance_numpy(v1, v2):
    '''
        compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)
        numpy ndarray version.
    '''
    v1_sq =  np.inner(v1, v1)
    v2_sq =  np.inner(v2, v2)
    dis = 1 - np.inner(v1, v2) / math.sqrt(v1_sq * v2_sq)
    return dis


def cosine_distance_tensor(v1, v2):
    '''
        compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)
        pytorch tensor version.
    '''
    v1_sq =  torch.dot(v1, v1)
    v2_sq =  torch.dot(v2, v2)
    dis = 1 - torch.dot(v1, v2) / (v1_sq * v2_sq).sqrt()
    return dis


def cosine_similarity_tensor(v1, v2):
    '''
        compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)
        pytorch tensor version.
    '''
    v1_sq =  torch.dot(v1, v1)
    v2_sq =  torch.dot(v2, v2)
    similarity = torch.dot(v1, v2) / (v1_sq * v2_sq).sqrt()
    return similarity