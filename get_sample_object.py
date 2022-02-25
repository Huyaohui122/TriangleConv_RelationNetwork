from __future__ import print_function
import torch.utils.data as data
import os
import re
import random
from new_pre import *
bu_filename = 'test_5010'
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

fileList = []
labelList = []
alldata_class_number = 10
support_class_number =5
support_train_shot = 5
support_test_shot = 15
index = 501

def _getfiles(dirPath):
    files = os.listdir(dirPath)
    ptn = re.compile('.*\.shp')
    dtn= re.compile('.*\.xml')
    for f in files:
        # isdir, call self
        if (os.path.isdir(dirPath + '\\' + f)):
            getfiles(dirPath + '\\' + f)
        # isfile, judge
        elif (os.path.isfile(dirPath + '\\' + f)):
            des = dtn.match(f)
            res = ptn.match(f)
            if (res != None and des == None):
                fileList.append(dirPath + '\\' + res.group())
        else:
            fileList.append(dirPath + '\\无效文件')
def getfiles(dirPath):
    _getfiles(dirPath)
    class_get = random.sample(fileList, alldata_class_number)
    return class_get


def get_sample_object(path_1 = './data/' + bu_filename + '.shp'):
    bu_shape = gpd.read_file(path_1, encode='utf-8')
    bu_use = copy.deepcopy(bu_shape)
    bu_mbr, bu_use = get_shape_mbr(bu_use)
    bu_use = get_shape_normalize_final(bu_use, hparams['if_scale_y'])
    if hparams['if_simplify']:
        bu_use = get_shape_simplify(bu_use, hparams['tor_dist'], hparams['tor_cos'], simplify_type=0)
    bu_use = reset_start_point(bu_use)
    bu_node = get_node_features(bu_use)
    bu_line = get_line_features_final(bu_node, hparams['seq_length'])
    bu_detail = get_inter_features(bu_line)
    bu_detail = get_neat_features(bu_detail, hparams['seq_length'], hparams['rotate_length'])
    all_data_x = np.array(bu_detail['xs'])
    train_x = all_data_x[:index * hparams['seq_length']]
    train_x = train_x.reshape(index * hparams['seq_length'], 1)
    all_data_y = np.array(bu_detail['ys'])
    train_y = all_data_y[:index * hparams['seq_length']]
    train_y = train_y.reshape(index * hparams['seq_length'], 1)
    train_x_y = np.concatenate((train_x, train_y), axis=1)

    train_xy_reshape = train_x_y.reshape(index, hparams['seq_length'], 2)

    return train_xy_reshape

def getdata():
    data_per_class_list = []
    path = 'split_5010'
    res = getfiles(path)
    for i in res:
        labelList.append(i.replace(".shp", ".txt"))
    for f in res:
        label = f.replace(".shp", ".csv")
        train_data_per_class = get_sample_object(path_1=str(f))
        data_per_class_list.append(train_data_per_class)
    return data_per_class_list,index
