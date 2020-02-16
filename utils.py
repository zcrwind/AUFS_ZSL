# -*- coding: utf-8 -*-

import numpy as np

class NormalNLLLoss(object):
    '''
    Calculate the negative log likelihood of normal distribution.
    This needs to be minimised.
    Treating Q(cj | x) as a factored Gaussian.

    zcr: see https://github.com/Natsu6767/InfoGAN-PyTorch/blob/master/utils.py for more details.
    '''
    def __call__(self, x, mu, var):
        
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll