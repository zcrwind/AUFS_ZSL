#!/bin/bash

mode='train'
dataset_name='cub'	# 'awa', 'awa2', 'apy', 'sun'

resume='pass'

n_iteration=5000
batch_size=64
lr_G=1e-3
lr_D=1e-3
lr_R=1e-3
weight_decay=1e-2
optimizer='adam'

labelIdxStart0or1=1
root_dir='../data'
save_dir='./checkpoint'
all_visualFea_label_file='res101.mat'
auxiliary_file='att_splits.mat'
use_z='true'
z_dim=100

gpuid=1

centroid_lambda=1
_lambda=0.00015
gp_lambda=10
regression_lambda=1

n_iter_D=1
n_iter_G=5

n_generation_perClass=50
classifier_type='softmax'
n_epoch_sftcls=100
use_pca='false'
reduced_dim_pca=1024

use_od='false'
miu=1.2			# for "adaptive outlier detection", miu >= 1
od_lambda=0.05	# the trade-off hyperparameter for cosine loss in "adaptive outlier detection"
MI_lambda=1e-6	# the trade-off hyperparameter for MMICC loss

python main.py \
	--mode ${mode} \
	--dataset_name ${dataset_name} \
	--resume ${resume} \
	--n_iteration ${n_iteration} \
	--batch_size ${batch_size} \
	--lr_G ${lr_G} \
	--lr_D ${lr_D} \
	--lr_R ${lr_R} \
	--weight_decay ${weight_decay} \
	--optimizer ${optimizer} \
	--labelIdxStart0or1 ${labelIdxStart0or1} \
	--root_dir ${root_dir} \
	--save_dir ${save_dir} \
	--all_visualFea_label_file ${all_visualFea_label_file} \
	--auxiliary_file ${auxiliary_file} \
	--use_z ${use_z} \
	--z_dim ${z_dim} \
	--gpuid ${gpuid} \
	--centroid_lambda ${centroid_lambda} \
	--_lambda ${_lambda} \
	--gp_lambda ${gp_lambda} \
	--regression_lambda ${regression_lambda} \
	--n_iter_D ${n_iter_D} \
	--n_iter_G ${n_iter_G} \
	--n_generation_perClass ${n_generation_perClass} \
	--classifier_type ${classifier_type} \
	--n_epoch_sftcls ${n_epoch_sftcls} \
	--use_pca ${use_pca} \
	--reduced_dim_pca ${reduced_dim_pca} \
	--use_od ${use_od} \
	--miu ${miu} \
	--od_lambda ${od_lambda} \
	--MI_lambda ${MI_lambda}
