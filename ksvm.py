# -*- coding: utf-8 -*-

'''
SVM algorithm
'''

import numpy as np
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

# from data_util import MyDataset


class MyKSVM(object):
    '''zcr svm'''
    def __init__(self, dataset_object):
        self.training_data  = dataset_object['tr_data']
        self.training_label = dataset_object['tr_label']
        self.test_data      = dataset_object['te_data']
        self.test_label     = dataset_object['te_label']
        self.num_test       = self.test_data.shape[0]


    def linear_kernel(self):
        '''linear kernel SVM'''
        svm_clf = Pipeline((
                ('scaler', StandardScaler()),
                ('linear_svc', svm.LinearSVC(C=1, loss='hinge')),
            ))
        svm_clf.fit(self.training_data, self.training_label)      # train svm

        predictions = svm_clf.predict(self.test_data)
        num_correct = sum(int(yhat == y) for yhat, y in zip(predictions, self.test_label))

        acc = num_correct / self.num_test
        print('[linear kernel] Acc is: %.4f' % acc)
        return acc


    def nonlinear_kernel(self):
        '''polynomial kernel'''
        polynomial_svm_clf = Pipeline((
                ('poly_features', PolynomialFeatures(degree=1)),
                ('scaler', StandardScaler()),
                ('svm_clf', svm.LinearSVC(C=10, loss='hinge')),
            ))
        polynomial_svm_clf.fit(self.training_data, self.training_label)

        predictions = polynomial_svm_clf.predict(self.test_data)
        num_correct = sum(int(yhat == y) for yhat, y in zip(predictions, self.test_label))

        acc = num_correct / self.num_test
        print('[nonlinear kernel] Acc is: %.4f' % acc)
        return acc


    def poly_kernel(self):
        '''poly kernel'''
        poly_kernel_svm_clf = Pipeline((
                ('scaler', StandardScaler()),
                ('svm_clf', svm.SVC(kernel='poly', degree=3, coef0=1, C=5)),
            ))

        poly_kernel_svm_clf.fit(self.training_data, self.training_label)

        predictions = poly_kernel_svm_clf.predict(self.test_data)
        num_correct = sum(int(yhat == y) for yhat, y in zip(predictions, self.test_label))

        acc = num_correct / self.num_test
        print('[poly kernel] Acc is: %.4f' % acc)
        return acc

    def RBF_kernel(self):
        '''Gaussian RBF Kernel'''
        rbf_kernel_svm_clf = Pipeline((
                ('scaler', StandardScaler()),
                ('svm_clf', svm.SVC(kernel='rbf', gamma=5, C=0.1)),     # C=0.01
            ))
        rbf_kernel_svm_clf.fit(self.training_data, self.training_label)
        predictions = rbf_kernel_svm_clf.predict(self.test_data)
        num_correct = sum(int(yhat == y) for yhat, y in zip(predictions, self.test_label))

        acc = num_correct / self.num_test
        print('[Gaussian RBF Kernel] Acc is: %.4f' % acc)
        return acc


if __name__ == "__main__":
    pass
    # usage:
    # data_dir = './'
    # data_file = 'train_data.mat'
    # dataset = MyDataset(data_dir, data_file)
    # ksvm = MyKSVM(dataset)

    # ksvm.linear_kernel()
    # ksvm.nonlinear_kernel()     # if degree == 3, may raise a MemoryError
    # ksvm.poly_kernel()
    # ksvm.RBF_kernel()
