import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
import numpy as np
import math
from datetime import datetime
import scipy as sp
import scipy.stats
from sklearn.metrics import classification_report, \
    confusion_matrix, \
    cohen_kappa_score, \
    precision_score, \
    accuracy_score
from torchvision import transforms
from common_usages import *
from IP_data_loader import IndianPines
from classifyUtils import plot_confusion_matrix
import matplotlib.pyplot as plt

GPU = 0
LEARNING_RATE = 0.0005
n_classes = 16
n_way = n_classes
n_shot = 5
n_query = 15
n_test_way = n_classes
n_test_shot = 40
dropout = 0.5


class ProtoNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, output_dim):
        super(ProtoNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(200, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64, momentum=0.01, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64, momentum=0.01, affine=True),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64, momentum=0.01, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64, momentum=0.01, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))

    def forward(self, x, training=False):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).mean(2)


def test(support, query, proto_network):
    sample_features = proto_network(Variable(torch.from_numpy(support)).cuda(GPU))  # 5x64
    sample_features = sample_features.view(n_test_way, n_test_shot, 64)
    sample_features = torch.mean(sample_features, 1).squeeze(1)
    test_features = proto_network(Variable(torch.from_numpy(query)).cuda(GPU))  # 20x64
    test_features = test_features.squeeze()
    dists = euclidean_dist(test_features, sample_features)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_test_way, len(query), -1)
    _, y_hat = log_p_y.max(2)
    predict_labels = torch.argmax(-dists, dim=1)
    return predict_labels.cpu().numpy()


def main():
    seed_everything()
    width, height, channel = 11, 11, 200
    zgff = IndianPines(width)
    name = "IP-allband2d"
    Xtest = np.load('./data/IPXtestWindowSize11testRatio0.95.npy')
    Xtrain = np.load('./data/IPXtrainWindowSize11testRatio0.95.npy')
    ytest = np.load('./data/IPytestWindowSize11testRatio0.95.npy')
    ytrain = np.load('./data/IPytrainWindowSize11testRatio0.95.npy')

    X_dict = {}
    for i in range(n_classes):
        X_dict[i] = []
    tmp_count_index = 0
    for _, y_index in enumerate(ytrain):
        y_i = int(y_index)
        if y_i in X_dict:
            X_dict[y_i].append(Xtrain[tmp_count_index])
        else:
            X_dict[y_i] = []
            X_dict[y_i].append(Xtrain[tmp_count_index])
        tmp_count_index += 1
    for i in range(n_classes):
        arr = np.array(X_dict[i])
        if len(arr) < 10:
            arr = np.tile(arr, (20))
        if len(arr) < 20:
            arr = np.tile(arr, (10))
        if len(arr) < 40:
            arr = np.tile(arr, (2))
        X_dict[i] = arr
    del Xtrain
    a = datetime.now()
    proto_network = ProtoNetwork(64)
    proto_network.cuda(GPU)
    support_test = np.zeros([n_test_way, n_test_shot, width, height, channel], dtype=np.float32)
    predict_dataset = np.zeros([len(ytest)], dtype=np.int32)
    epi_classes = np.arange(n_classes)
    for i, epi_cls in enumerate(epi_classes):
        selected = np.random.permutation(len(X_dict[epi_cls]))[:n_test_shot]
        support_test[i] = zgff.getPatches(np.array(X_dict[epi_cls])[selected])
    support_test = support_test.transpose((0, 1, 4, 2, 3))
    support_test = np.reshape(support_test, [n_test_way * n_test_shot, channel, width, height])
    save_path = './model/IP/IP_proto_2d-5shotALLband.pth'.format(n_way, n_shot)
    proto_network.load_state_dict(torch.load(save_path))
    print('Testing...')
    proto_network.eval()
    del X_dict
    test_num = 3000
    test_count = int(len(ytest) / test_num)
    for i in range(test_count):
        query_test = zgff.getPatches(Xtest[i * test_num:(i + 1) * test_num]).astype(np.float32)
        query_test = np.reshape(query_test, [-1, width, height, channel])
        query_test = query_test.transpose((0, 3, 1, 2))
        predict_dataset[i * test_num:(i + 1) * test_num] = test(support_test, query_test, proto_network)
    query_test = zgff.getPatches(Xtest[test_count * test_num:]).astype(np.float32)
    query_test = np.reshape(query_test, [-1, width, height, channel])
    query_test = query_test.transpose((0, 3, 1, 2))
    del Xtest
    predict_dataset[test_count * test_num:] = test(support_test, query_test, proto_network)
    confusion = confusion_matrix(ytest, predict_dataset)
    acc_for_each_class = precision_score(ytest, predict_dataset, average=None)
    average_accuracy = np.mean(acc_for_each_class)
    kappa = cohen_kappa_score(ytest, predict_dataset)
    overall_acc = accuracy_score(ytest, predict_dataset)
    print('OA: {:.2f}'.format(overall_acc * 100))
    print('kappa:{:.4f}'.format(kappa))
    print('PA:')
    for i in range(len(acc_for_each_class)):
        print('{:.2f}'.format(acc_for_each_class[i] * 100))
    print('AA: {:.2f}'.format(average_accuracy * 100))

    target_name = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J ', 'K', 'L', 'M', 'N', 'O', 'P', ]
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(confusion, classes=target_name, normalize=False, title='', cmap=plt.cm.Blues)
    # plt.show()
    #plt.matshow(confusion, cmap=plt.cm.Blues)
    plt.savefig('./map/{}_confusion_matrix.png'.format(name))


if __name__ == '__main__':
    main()