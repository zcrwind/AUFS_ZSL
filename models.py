# -*- coding: utf-8 -*-


'''models of our Adversarial Unseen Feature Synthesis (AUFS)'''

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class _netG(nn.Module):
	'''generator'''
	def __init__(self, se_fea_dim, vi_fea_dim, z_dim):
		super(_netG, self).__init__()
		self.main = nn.Sequential(nn.Linear((z_dim + se_fea_dim), 2048),
                                  nn.LeakyReLU(),

                                  nn.Linear(2048, 4096),
                                  # nn.BatchNorm1d(4096),
                                  nn.LeakyReLU(),

                                  nn.Linear(4096, vi_fea_dim),
                                  nn.Tanh())

	def forward(self, se_fea, z):
		_input = torch.cat([z, se_fea], 1)
		output = self.main(_input)
		return output

class _netG2(nn.Module):
	'''generator'''
	def __init__(self, se_fea_dim, vi_fea_dim):
		super(_netG2, self).__init__()
		self.main = nn.Sequential(nn.Linear((se_fea_dim), 1024),
								  nn.BatchNorm1d(1024),
                                  nn.LeakyReLU(),

                                  nn.Linear(1024, 2048),
                                  nn.BatchNorm1d(2048),
                                  nn.LeakyReLU(),

                                  nn.Linear(2048, 4096),
                                  nn.BatchNorm1d(4096),
                                  nn.LeakyReLU(),

                                  nn.Linear(4096, vi_fea_dim),
                                  )
                                  # nn.Tanh())

	def forward(self, se_fea):
		output = self.main(se_fea)
		return output


class _netD_backup(nn.Module):
	'''
		discriminator
		n_class: number of the classes for auxiliary classification.
	'''
	def __init__(self, vi_fea_dim, n_class=51):
		super(_netD, self).__init__()
		self.D_shared = nn.Sequential(nn.Linear(vi_fea_dim, 4096),
									  nn.ReLU(),
									  nn.Linear(4096, 2048),
									  nn.ReLU())

		self.D_gan = nn.Linear(2048, 1)					# GAN loss
		self.D_classifier = nn.Linear(2048, n_class)	# auxiliary classification loss


	def forward(self, vi_feature):
		hidden = self.D_shared(vi_feature)
		gan_loss = self.D_gan(hidden)
		classification_loss = self.D_classifier(hidden)
		return gan_loss, classification_loss
		

class _netD(nn.Module):
	'''
		discriminator
		n_class: number of the classes for auxiliary classification.
	'''
	def __init__(self, vi_fea_dim, n_class):
		super(_netD, self).__init__()
		self.D_shared = nn.Sequential(nn.Linear(vi_fea_dim, 4096),
									  nn.ReLU(),
									  nn.Linear(4096, 1024),
									  nn.ReLU())

		self.D_gan = nn.Linear(1024, 1)					# GAN loss
		self.D_classifier = nn.Linear(1024, n_class)	# auxiliary classification loss


	def forward(self, vi_feature):
		hidden = self.D_shared(vi_feature)
		gan_loss = self.D_gan(hidden)
		classification_loss = self.D_classifier(hidden)
		return hidden, gan_loss, classification_loss


class Regressor(nn.Module):
	'''
		regressor for generated visual feature -> semantic feature (e.g., word vector or attribute)
		Args:
			input_dim:  the dimension of generated visual feature (e.g., 2048 for resnet101 feature)
			output_dim: the dimension of semantic feature (e.g., 312 for CUB attribute)
	'''
	def __init__(self, input_dim, output_dim):
		super(Regressor, self).__init__()

		self.model = nn.Sequential(
			nn.Linear(input_dim, 4096),
			nn.LeakyReLU(),
			# nn.Linear(2048, 4096),
			# nn.Tanh(),
			nn.Linear(4096, output_dim),
		)

	def forward(self, x):
		x = self.model(x)
		return x


class _netQ(nn.Module):
	'''
	Q head of the discriminator, for calculating the mutual information between visual-semantic pairs.
	'''

	def __init__(self, input_dim=1024):
		super(_netQ, self).__init__()
		self.mu_module = nn.Linear(input_dim, 1)
		self.var_module = nn.Linear(input_dim, 1)

	def forward(self, output_of_D_shared):
		mu = self.mu_module(output_of_D_shared)
		var = torch.exp(self.var_module(output_of_D_shared))
		return mu, var


if __name__ == '__main__':
	vi_fea_dim = 8192
	se_fea_dim = 300
	z_dim = 100
	netG = _netG(se_fea_dim, vi_fea_dim, z_dim)
	netD = _netD(vi_fea_dim)
	print(netG)
	print(netD)
