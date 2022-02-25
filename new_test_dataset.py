from __future__ import print_function
import torch.utils.data as data
from new_pre import *
import os
import re
import random
import torch
hparams = {
    'if_simplify': True
    , 'tor_dist': 0.1
    , 'tor_cos': 0.99
    , 'if_scale_y': False

    , 'scale_type': 'final'
    , 'seq_length':  16
    , 'rotate_type': 'equal'
    , 'rotate_length': 1
    , 'norm_type': 'minmax'
}

train_dataset_list = []
test_dataset_list = []
fileList = []
labelList = []
class_number = 10
support_number =5

support_train_shot = 5
support_test_shot = 15






class Model10DataSet(data.Dataset):
    def __init__(self, train=True, train_xy_reshape = None, index =None, shot = 5,test_shot = 15, label_number = 0):

        np.random.shuffle(train_xy_reshape)

        new_label = np.full(index, label_number, dtype= np.int64)
        label_train =new_label[:index]
        label_train =label_train.reshape(-1,1)

        if train:
            modelnet_data = np.zeros([0, hparams['seq_length'], 2], dtype=np.float64)
            modelnet_label = np.zeros([0, 1], np.float64)
            modelnet_data = np.concatenate([modelnet_data, train_xy_reshape], axis=0)
            modelnet_data_split = modelnet_data[0:shot,:,:]

            modelnet_label = np.concatenate([modelnet_label, label_train], axis=0)
            modelnet_label_split = modelnet_label[0:shot,:]
        else:
            modelnet_data = np.zeros([0, hparams['seq_length'], 2], dtype=np.float64)

            modelnet_data = np.concatenate([modelnet_data, train_xy_reshape], axis=0)
            modelnet_data_split = modelnet_data[shot:shot+test_shot, :, :]
            modelnet_label = label_train
            modelnet_label_split = modelnet_label[shot:shot+test_shot,:]

        self.point_cloud = modelnet_data_split
        self.label = modelnet_label_split
    def __getitem__(self, item):
        return self.point_cloud[item], self.label[item]

    def __len__(self):
        return self.label.shape[0]




if __name__ == '__main__':

    support_train, support_train_labels, support_test, support_test_labels = Model10DataSet()
    print(support_train.shape,support_train_labels.shape,support_test.shape,support_test_labels.shape)
