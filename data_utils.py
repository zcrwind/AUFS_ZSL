# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pickle
from scipy import io as sio
from sklearn.decomposition import PCA

from torch.utils.data import Dataset
from torchvision import transforms
import torch.utils.data as data


'''
use data of the "LearningToCompare" paper.


1. CUB:
att_splits.mat
    allclasses_names       200 x 1
    att                    312 x 200
    test_seen_loc         1764 x 1  [use]
    test_unseen_loc       2967 x 1  [use]
    train_loc             5875 x 1
    trainval_loc          7057 x 1  [use]
    val_loc               2946 x 1

<check: 7057(trainval_loc) + 1764(test_seen_loc) + 2967(test_unseen_loc) = 11788>

res101.mat:
    features              2048 x 11788
    labels               11788 x 1

2. AwA:
att_splits.mat:
    allclasses_names        50 x 1
    att                     85 x 50
    test_seen_loc         4958 x 1  [use]
    test_unseen_loc       5685 x 1  [use]
    train_loc            16864 x 1
    trainval_loc         19832 x 1  [use]
    val_loc               7926 x 1

<check: 19832(trainval_loc) + 4958(test_seen_loc) + 5685(test_unseen_loc) = 30475>

res101.mat
    features              2048 x 30475
    image_files          30475 x 1
    labels               30475 x 1

3. AwA2:
    
<check: 23527(trainval_loc) + 5882(test_seen_loc) + 7913(test_unseen_loc) = 37322>


4. SUN:

<check: 10320(trainval_loc) + 2580(test_seen_loc) + 1440(test_unseen_loc) = 14340>


5. aPY:

<check: 5932(trainval_loc) + 1483(test_seen_loc) + 7924(test_unseen_loc) = 15339>


'''



class ZSL_Dataset(Dataset):
    '''
        Args:
            root_dir: pass
            mode: `train` or `test`
            all_prototype_semantic_feature_file: prototype semantic feature of `n_class`, e.g., for CUB dataset, there are 200 prototype semantic feature.
            auxiliary_file: contains split file and semantic feature.
            use_pca: use PCA for visual feature or not.
            reduced_dim_pca: the dim of visual feature after PCA.
    '''

    def __init__(self, root_dir, dataset_name, mode, all_visualFea_label_file, auxiliary_file, use_pca=False, reduced_dim_pca=None):
        super(ZSL_Dataset, self).__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.dataset = dataset_name

        if dataset_name.lower() == 'cub':
            dataset_subdir = 'CUB'
        elif dataset_name.lower() == 'awa':
            dataset_subdir = 'AwA'
        elif dataset_name.lower() == 'awa2':
            dataset_subdir = 'AwA2'
        elif dataset_name.lower() == 'apy':
            dataset_subdir = 'APY'
        elif dataset_name.lower() == 'sun':
            dataset_subdir = 'SUN'
        else:
            raise RuntimeError('Unknown dataset of "%s"' % dataset_name)

        # visual feature for whole dataset
        all_visFea_label_path = os.path.join(root_dir, dataset_subdir, all_visualFea_label_file)
        all_visFea_label_data = sio.loadmat(all_visFea_label_path)
        all_vis_fea = all_visFea_label_data['features'].astype(np.float32).T

        ## prepare label ##
        all_labels = all_visFea_label_data['labels'].astype(np.long).squeeze() - 1   # from 1-based to 0-based

        auxiliary_path = os.path.join(root_dir, dataset_subdir, auxiliary_file)
        auxiliary_data = sio.loadmat(auxiliary_path)
        tr_vis_fea_idx = auxiliary_data['trainval_loc'].squeeze() - 1
        te_vis_fea_idx_seen = auxiliary_data['test_seen_loc'].squeeze() - 1
        te_vis_fea_idx_unseen = auxiliary_data['test_unseen_loc'].squeeze() - 1
        all_prototype_semantic_feature = auxiliary_data['att'].astype(np.float32).T

        ## prepare training data ##
        self.tr_vis_fea = all_vis_fea[tr_vis_fea_idx]
        self.tr_label = all_labels[tr_vis_fea_idx]
        self.tr_sem_fea = all_prototype_semantic_feature[self.tr_label]
        self.tr_labelID = np.unique(self.tr_label)                  # label set
        self.n_tr_class = self.tr_labelID.shape[0]
        self.tr_sem_fea_pro = all_prototype_semantic_feature[self.tr_labelID]
        assert self.tr_vis_fea.shape[0] == self.tr_sem_fea.shape[0] == self.tr_label.shape[0]

        ## prepare test data ##
        # unseen
        self.te_vis_fea_unseen = all_vis_fea[te_vis_fea_idx_unseen]
        self.te_label_unseen = all_labels[te_vis_fea_idx_unseen]
        self.te_sem_fea_unseen = all_prototype_semantic_feature[self.te_label_unseen]
        self.te_labelID_unseen = np.unique(self.te_label_unseen)    # label set
        self.te_sem_fea_pro_unseen = all_prototype_semantic_feature[self.te_labelID_unseen]
        assert self.te_vis_fea_unseen.shape[0] == self.te_label_unseen.shape[0] == self.te_sem_fea_unseen.shape[0]
        assert self.te_labelID_unseen.shape[0] == self.te_sem_fea_pro_unseen.shape[0]
        # seen
        self.te_vis_fea_seen = all_vis_fea[te_vis_fea_idx_seen]
        self.te_label_seen = all_labels[te_vis_fea_idx_seen]
        self.te_sem_fea_seen = all_prototype_semantic_feature[self.te_label_seen]
        self.te_labelID_seen = np.unique(self.te_label_seen)
        self.te_sem_fea_pro_seen = all_prototype_semantic_feature[self.te_labelID_seen]
        assert self.te_vis_fea_seen.shape[0] == self.te_label_seen.shape[0] == self.te_sem_fea_seen.shape[0]
        assert self.te_labelID_seen.shape[0] == self.te_sem_fea_pro_seen.shape[0]


        tr_vis_fea = self.tr_vis_fea
        te_vis_fea_unseen = self.te_vis_fea_unseen
        te_vis_fea_seen = self.te_vis_fea_seen
        if use_pca == 'true':
            n_components = reduced_dim_pca  # the dim of visual feature after PCA
            if n_components < tr_vis_fea.shape[1]:
                raise(RuntimeError('visual feature dim < the dim that PCA will reduced to!'))
            tr_vis_fea = pca(n_components, tr_vis_fea)
            te_vis_fea_unseen = pca(n_components, te_vis_fea_unseen)
            te_vis_fea_seen = pca(n_components, te_vis_fea_seen)
        tr_vis_fea = (tr_vis_fea - tr_vis_fea.mean()) / tr_vis_fea.var()
        te_vis_fea_unseen = (te_vis_fea_unseen - te_vis_fea_unseen.mean()) / te_vis_fea_unseen.var()
        te_vis_fea_seen = (te_vis_fea_seen - te_vis_fea_seen.mean()) / te_vis_fea_seen.var()
        self.tr_vis_fea = tr_vis_fea
        self.te_vis_fea_unseen = te_vis_fea_unseen
        self.te_vis_fea_seen = te_vis_fea_seen

        self.all_prototype_semantic_feature = all_prototype_semantic_feature
        self.vis_fea_dim = self.tr_vis_fea.shape[1]
        self.sem_fea_dim = self.all_prototype_semantic_feature.shape[1]

        # the function of the self-defined maps: map [3, 6, 17, ..., 199] to [0, 1, 2, ..., 150] (take CUB(150/50 split) as an example)
        self.tr_classIdx_map = dict(zip(self.tr_labelID, range(len(self.tr_labelID))))
        self.te_classIdx_unseen_map = dict(zip(self.te_labelID_unseen, range(len(self.te_labelID_unseen))))
        self.te_classIdx_seen_map = dict(zip(self.te_labelID_seen, range(len(self.te_labelID_seen))))
 

    def __getitem__(self, index):
        if self.mode == 'train':
            vis_fea = self.tr_vis_fea[index]
            sem_fea = self.tr_sem_fea[index]
            label = self.tr_classIdx_map[self.tr_label[index]]
            return vis_fea, sem_fea, label
        elif self.mode == 'test':
            vis_fea_unseen = self.te_vis_fea_unseen[index]
            sem_fea_unseen = self.te_sem_fea_unseen[index]
            label_unseen = self.te_classIdx_unseen_map[self.te_label_unseen[index]]
            vis_fea_seen = self.te_vis_fea_seen[index]
            sem_fea_seen = self.te_sem_fea_seen[index]
            label_seen = self.te_classIdx_seen_map[self.te_label_seen[index]]
            return vis_fea_unseen, sem_fea_unseen, label_unseen, vis_fea_seen, sem_fea_seen, label_seen


    def __len__(self):
        if self.mode == 'train':
            return self.tr_vis_fea.shape[0]
        elif self.mode == 'test':
            return (self.te_vis_fea_unseen.shape[0], self.te_vis_fea_seen.shape[0])


    def get_testData(self):
        te_unseen = (self.te_vis_fea_unseen, self.te_sem_fea_unseen, self.te_label_unseen, self.te_labelID_unseen, self.te_sem_fea_pro_unseen)
        te_seen   = (self.te_vis_fea_seen, self.te_sem_fea_seen, self.te_label_seen, self.te_labelID_seen, self.te_sem_fea_pro_seen)
        return te_unseen, te_seen

    def get_trainData(self):
        return self.tr_vis_fea, self.tr_sem_fea, self.tr_label, self.tr_labelID, self.tr_sem_fea_pro


    def get_tr_centroid(self):
        '''get the centroid of each class in training set.'''
        tr_cls_centroid = np.zeros([self.n_tr_class, self.tr_vis_fea.shape[1]]).astype(np.float32)
        for i in range(self.n_tr_class):
            current_tr_classId = self.tr_labelID[i]
            tr_cls_centroid[i] = np.mean(self.tr_vis_fea[self.tr_label == current_tr_classId], axis=0)
        return tr_cls_centroid


def pca(n_components, data):
    '''PCA for visual feature dimension reduction.'''
    pca = PCA(n_components=n_components)
    pca.fit(data)
    data_new = pca.transform(data)
    print('data.shape after PCA:', data_new.shape)
    return data_new



if __name__ == '__main__':
    root_dir = '../data'
    all_visualFea_label_file = 'res101.mat'
    auxiliary_file = 'att_splits.mat'

    dataset_name = 'cub'
    # dataset_name = 'awa'
    # dataset_name = 'awa2'
    # dataset_name = 'apy'
    # dataset_name = 'sun'
    mode = 'train'
    zsl_dataset = ZSL_Dataset(root_dir, dataset_name, mode, all_visualFea_label_file, auxiliary_file)
    zsl_dataloader = data.DataLoader(zsl_dataset, batch_size=64, shuffle=False, num_workers=2)
    for step, (vis_fea, sem_fea, label) in enumerate(zsl_dataloader):
        print('vis_fea.shape, sem_fea.shape, label.shape', vis_fea.shape, sem_fea.shape, label.shape)

