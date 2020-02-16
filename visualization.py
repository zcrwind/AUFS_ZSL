# -*- coding: utf-8 -*-


import os
import numpy as np
from tsne_python.tsne import tsne
import matplotlib
matplotlib.use('Agg')   # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

from data_utils import ZSL_Dataset


def visualization(feature, label, save_dir, nameStr):
    '''t-SNE visualization for visual features'''
    assert feature.shape[0] == label.shape[0]
    X = feature
    labels = label
    Y = tsne(X, 2, 50, 20.0)
    plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
    save_path = os.path.join(save_dir, nameStr + '.png')
    plt.savefig(save_path)
    print('visualization results are saved done in %s!' % save_dir)



if __name__ == '__main__':
    root_dir = '../data'
    dataset_name = 'awa'
    mode = 'train'
    all_visualFea_label_file = 'res101.mat'
    auxiliary_file='original_att_splits.mat'
    use_pca = 'false'
    reduced_dim_pca = None

    zsl_dataset = ZSL_Dataset(root_dir, dataset_name, mode, all_visualFea_label_file, auxiliary_file, use_pca, reduced_dim_pca)

    te_data_unseen, te_data_seen = zsl_dataset.get_testData()
    te_vis_fea_unseen, te_sem_fea_unseen, te_label_unseen, te_labelID_unseen, te_sem_fea_pro_unseen = te_data_unseen
    te_vis_fea_seen, te_sem_fea_seen, te_label_seen, te_labelID_seen, te_sem_fea_pro_seen = te_data_seen

    save_rootdir = './visualization'
    save_dir = os.path.join(save_rootdir, dataset_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    nameStr = 'unseen_' + dataset_name + '_' + all_visualFea_label_file.split('.')[0] + '_realFea'
    print('te_vis_fea_unseen.shape', te_vis_fea_unseen.shape)
    print('te_label_unseen.shape', te_label_unseen.shape)
    visualization(te_vis_fea_unseen, te_label_unseen, save_dir, nameStr)

    