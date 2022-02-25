import torch
import torch.nn as nn
import os
import numpy as np
import re
import random
import math
from torch.optim import lr_scheduler
from collections import OrderedDict
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F
import torch.utils.data as data
from get_sample_object import getdata
from new_test_dataset import Model10DataSet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
dirpath: str, the path of the directory
"""
def conv_bn_block(input, output, kernel_size):
    return nn.Sequential(
        nn.Conv1d(input, output, kernel_size),
        nn.BatchNorm1d(output),
        nn.ReLU(inplace=True)
    )


def fc_bn_block(input, output):
    return nn.Sequential(
        nn.Linear(input, output),
        nn.BatchNorm1d(output),
        nn.ReLU(inplace=True)
    )

class TriangleConv(nn.Module):
    def __init__(self, layers):
        super(TriangleConv, self).__init__()
        self.layers = layers
        mlp_layers = OrderedDict()
        for i in range(len(self.layers) - 1):
            if i == 0:
                mlp_layers['conv_bn_block_{}'.format(i + 1)] = conv_bn_block(4 * self.layers[i], self.layers[i + 1], 1)
            else:
                mlp_layers['conv_bn_block_{}'.format(i + 1)] = conv_bn_block(self.layers[i], self.layers[i + 1], 1)
        self.mlp = nn.Sequential(mlp_layers)



    def forward(self, X):
        B, N, F = X.shape
        k_indexes = []
        for i in range(N):
            if i == 0:
                k_indexes.append([N - 1, i + 1])
            elif i == N-1:
                k_indexes.append([i - 1, 0])
            else:
                k_indexes.append([i - 1, i+1])
        k_indexes_tensor = torch.Tensor(k_indexes)
        k_indexes_tensor = k_indexes_tensor.long()
        x1 = torch.zeros(B, N, 2, F).to(device)
        for idx, x in enumerate(X):
            x1[idx] = x[k_indexes_tensor]
        x2 = X.reshape([B, N, 1, F]).float()
        x2 = x2.expand(B, N, 2, F)
        x2 = x2-x1
        x3 = x2[:, :, 0:1, :]
        x4 = x2[:, :, 1:2, :]
        x4 = x3-x4
        x5 = X.reshape([B, N, 1, F]).float()
        x2 = x2.reshape([B, N, 1, 2*F])
        x_triangle = torch.cat([x5, x2, x4], dim=3)
        x_triangle=torch.squeeze(x_triangle)
        x_triangle = x_triangle.permute(0, 2, 1)
        x_triangle = torch.tensor(x_triangle,dtype=torch.float32).to(device)
        out = self.mlp(x_triangle)
        out = out.permute(0, 2, 1)
        return out


class DPCN_vanilla(nn.Module):
    def __init__(self):
        super(DPCN_vanilla, self).__init__()

        #self.num_classes = num_classes
        self.triangleconv_1 = TriangleConv(layers=[2, 64, 64, 64])
        self.triangleconv_2 = TriangleConv(layers=[64, 1024,64])
        #self.fc_block_4 = fc_bn_block(1024, 512)
        #self.drop_4 = nn.Dropout(0.5)
        #self.fc_block_5 = fc_bn_block(512, 256)
        #self.drop_5 = nn.Dropout(0.5)
        #self.fc_6 = nn.Linear(256, 64)
        #self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, N, C = x.shape
        assert C == 2, 'dimension of x does not match'
        x = self.triangleconv_1(x)
        x = self.triangleconv_2(x)
        x = x.permute(0, 2, 1)
        x = nn.MaxPool1d(N)(x)
        x = x.reshape([B, -1])

        return x

class RelationNetwork(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv1d(2,128,kernel_size=3),
                        nn.BatchNorm1d(128,momentum=0.9, affine=True),
                        nn.LeakyReLU(),
                        nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv1d(128,64,kernel_size=3),
                        nn.BatchNorm1d(64,momentum=0.9, affine=True),
                        nn.LeakyReLU(),
                        nn.MaxPool1d(2))
        self.fc1 = nn.Linear(896,8)#896
        self.fc2 = nn.Linear(8,1)



    def forward(self,x):
        out = self.layer1(x)#
        out = self.layer2(out)#
        out = out.view(out.size(0),-1)#
        out = F.relu(self.fc1(out))#

        out = torch.sigmoid(self.fc2(out))#
        return out


class_number = 2
support_number = 5
support_train_shot = 5
support_test_shot = 15


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0]  * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


def new_getdataset(support_data = None, val_data = None, index = None, train_shuffle=True, test_shuffle=True, train1= True):
    support_train_dataset_list = []
    support_test_dataset_list = []
    label_number = 0
    if train1 == True:
        per_task_data = random.sample(support_data,class_number)
    else:
        per_task_data = random.sample(val_data, class_number)
    for i in per_task_data:

        support_train_perclass_dataset = Model10DataSet(train=True,train_xy_reshape = i, index =index,
                                                        shot = support_train_shot,test_shot = support_test_shot,
                                                        label_number = label_number)
        support_train_dataset_list.append(support_train_perclass_dataset)

        support_test_perclass_dataset = Model10DataSet(train=False, train_xy_reshape= i, index=index,
                                                        shot=support_train_shot, test_shot=support_test_shot,
                                                        label_number=label_number)
        support_test_dataset_list.append(support_test_perclass_dataset)
        label_number =label_number + 1
    train_dataset_concat = data.ConcatDataset(support_train_dataset_list)
    test_dataset_concat = data.ConcatDataset(support_test_dataset_list)
    train_loader = torch.utils.data.DataLoader(train_dataset_concat, batch_size=class_number * support_train_shot,
                                               shuffle=train_shuffle, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset_concat, batch_size=class_number * support_test_shot,
                                              shuffle=test_shuffle, num_workers=0)
    support_train, support_train_labels = train_loader.__iter__().next()  # 25,25
    # print(support_train.shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    support_train, support_train_labels = support_train.to(device), support_train_labels.to(device)
    support_test, support_test_labels = test_loader.__iter__().next()  # 75,75
    support_test, support_test_labels = support_test.to(device), support_test_labels.to(device)
    # print(type(train_loader))
    return support_train, support_train_labels, support_test, support_test_labels

def main():


    print("init data:")

    all_train_data, index = getdata()
    all_train_data = np.array(all_train_data)
    print("This is all_train_data:", all_train_data.shape)
    support_data = all_train_data[0:support_number, :, :, :].tolist()
    val_data = all_train_data[support_number:, :, :, :].tolist()
    print("all_train_data:", np.array(all_train_data).shape)
    print("support_data:", np.array(support_data).shape)
    print("val_data:", np.array(val_data).shape)


    print("init network")
    model = DPCN_vanilla().cuda()
    relation_network = RelationNetwork(16, 8)
    relation_network.apply(weights_init)
    relation_network.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    schedular = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=0.001)
    relation_network_scheduler = lr_scheduler.StepLR(relation_network_optim, step_size=10000, gamma=0.5)

    print("Training~~~~~")
    best_accuracy = 0.0
    for episode in range(20000):
        schedular.step()
        relation_network_scheduler.step()
        model.train()
        support_train, support_train_labels, support_test, support_test_labels = new_getdataset(support_data = support_data, index = index, val_data = val_data, train_shuffle=False,test_shuffle=True, train1=True)
        out_support_train = model(support_train)
        out_support_train_2 = out_support_train.view(class_number, support_train_shot, 1, 64)
        out_support_train_3 = torch.mean(out_support_train_2, 1).squeeze(1)
        out_support_train_repeat = out_support_train_3.unsqueeze(0).repeat(class_number * support_test_shot, 1, 1)


        out_support_test = model(support_test)
        out_support_test_repeat = out_support_test.unsqueeze(0).repeat(class_number, 1, 1)
        out_support_test_repeat_transpose = torch.transpose(out_support_test_repeat, 0, 1)

        relation_pairs = torch.cat((out_support_train_repeat, out_support_test_repeat_transpose), 2).view(-1,  1* 2, 64)
        relations = relation_network(relation_pairs).view(-1, class_number)

        mse = nn.MSELoss().cuda()
        input_zero = torch.zeros(support_test_shot * class_number, class_number).cuda()
        support_test_labels = torch.squeeze(support_test_labels, dim=-1)

        input_scatter = input_zero.scatter_(1, support_test_labels.long().view(-1, 1),1)
        one_hot_labels = Variable(input_scatter).cuda()

        loss = mse(relations, one_hot_labels)

        optimizer.zero_grad()
        relation_network_optim.zero_grad()
        loss.backward()
        optimizer.step()
        relation_network_optim.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(relation_network.parameters(), 0.5)
        if (episode + 1) % 100 == 0:
            print("episode:", episode + 1, "loss", loss.item())
        if(episode+1)%500== 0:
            print("Testing")
            total_rewards = 0

            for i in range(300):
                test_train, test_train_labels, test_test, test_test_labels = new_getdataset(support_data = support_data, val_data = val_data, index = index, train_shuffle=False,test_shuffle=True, train1= False)
                out_test_train = model(test_train)
                out_test_train_2 = out_test_train.view(class_number, support_train_shot, 1, 64)
                out_test_train_3 = torch.mean(out_test_train_2, 1).squeeze(1)
                out_test_train_repeat = out_test_train_3.unsqueeze(0).repeat(class_number * support_test_shot,
                                                                                   1, 1)


                out_test_test = model(test_test)
                out_test_test_repeat = out_test_test.unsqueeze(0).repeat(class_number,  1, 1)
                out_test_test_repeat_transpose = torch.transpose(out_test_test_repeat, 0, 1)
                relation_pairs = torch.cat((out_test_train_repeat, out_test_test_repeat_transpose), 2).view(-1,1 * 2, 64)
                relations = relation_network(relation_pairs).view(-1, class_number)

                _, predict_labels = torch.max(relations.data, 1)
                predict_labels = predict_labels.cpu().int()

                test_test_labels = test_test_labels.cpu().int()
                rewards = [1 if predict_labels[j] == test_test_labels[j] else 0 for j in
                           range(class_number * support_test_shot)]

                total_rewards += np.sum(rewards)

            test_accuracy = total_rewards / 1.0 / class_number / support_test_shot / 300
            print("test accuracy:", test_accuracy)
            if test_accuracy > best_accuracy:

                # save networks
                #torch.save(model.state_dict(),str("./save_models/new_relation/model_" + str(class_number) +"way_" + str(support_test_shot) +"shot.pkl"))
                #torch.save(relation_network.state_dict(),str("./save_models/new_relation/relation_network_"+ str(class_number) +"way_" + str(support_test_shot) +"shot.pkl"))

                #print("save networks for episode:",episode+1)

                best_accuracy = test_accuracy
    print("best_accuracy:",best_accuracy)
    return best_accuracy
if __name__ == '__main__':
    main()


