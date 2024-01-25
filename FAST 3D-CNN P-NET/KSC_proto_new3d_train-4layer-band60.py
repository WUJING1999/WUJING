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
from KSC_data_loaderBS60 import KSC


GPU = 0
LEARNING_RATE = 0.0005
n_epochs = 60
n_episodes = 100
n_classes = 13
n_way = n_classes
n_shot = 5
n_query = 10
n_test_way = n_classes
n_test_shot = 40


def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test


class ProtoNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, output_dim):
        super(ProtoNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(7, 3, 3), stride=(2, 1, 1), padding=0),
            nn.ReLU(),
            nn.BatchNorm3d(num_features=64),
            #nn.MaxPool3d((2, 1, 1))
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(7, 3, 3), stride=(2, 1, 1), padding=0),
            nn.ReLU(),
            nn.BatchNorm3d(num_features=64),
            #nn.MaxPool3d((2, 1, 1))
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(7, 3, 3), stride=(2, 1, 1), padding=0),
            nn.ReLU(),
            nn.BatchNorm3d(num_features=64),
            #nn.MaxPool3d((2, 1, 1))
        )
        self.layer4 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(7, 3, 3), stride=(2, 1, 1), padding=0),
            nn.ReLU(),
            nn.BatchNorm3d(num_features=64),
            # nn.MaxPool3d((2, 1, 1))
        )
#
        #self.dropout1 = nn.Dropout(self.dropout)
        self.fc1 = nn.Sequential(
            nn.Linear(50 * 32, 32),
            nn.ReLU())
        #self.dropout2 = nn.Dropout(self.dropout)
        self.fc2 = nn.Sequential(
            nn.Linear(32, output_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        out = self.layer3(x)
        # print(out.shape)
        #out = self.dropout1(x)
        out = out.view(out.size(0), -1)
        #out = self.fc1(out)
        # = self.dropout2(out)
        #out = self.fc2(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


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
    sample_features = sample_features.view(n_test_way, n_test_shot, 64*75)
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
    width, height, channel = 11, 11, 60
    zgff = KSC(width)
    Xval = np.load('./data/KCXtestWindowSize11testRatio0.95.npy')
    Xtrain = np.load('./data/KCXtrainWindowSize11testRatio0.95.npy')
    yval = np.load('./data/KCytestWindowSize11testRatio0.95.npy')
    ytrain = np.load('./data/KCytrainWindowSize11testRatio0.95.npy')

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
    #proto_network = ProtoNetwork(64)
    proto_network = ProtoNetwork(64)
    proto_network.cuda(GPU)
    proto_network_optim = torch.optim.Adam(proto_network.parameters(), lr=LEARNING_RATE)
    proto_network_scheduler = StepLR(proto_network_optim, step_size=2000, gamma=0.5)
    save_path = './model/KSC/KC_proto_3d-5shotband60layer3-64output.pth'.format(n_way, n_shot)
    proto_network.train()
    last_epoch_loss_avrg = 0.
    last_epoch_acc_avrg = 0.
    best_acc = 0.

    support_test = np.zeros([n_test_way, n_test_shot, width, height, channel], dtype=np.float32)
    predict_dataset = np.zeros([len(yval)], dtype=np.int32)
    epi_classes = np.arange(n_classes)
    for i, epi_cls in enumerate(epi_classes):
        selected = np.random.permutation(len(X_dict[epi_cls]))[:n_test_shot]
        support_test[i] = zgff.getPatches(np.array(X_dict[epi_cls])[selected])
    support_test = support_test.transpose((0, 1, 4, 2, 3))
    support_test = np.reshape(support_test, [n_test_way * n_test_shot, 1, channel, width, height])
    for ep in range(n_epochs):
        proto_network.train()
        last_epoch_loss_avrg = 0.
        last_epoch_acc_avrg = 0.
        for epi in range(n_episodes):
            epi_classes = np.arange(n_way)
            # epi_classes = np.random.permutation(n_classes)[:n_way]
            samples = np.zeros([n_way, n_shot, width, height, channel], dtype=np.float32)
            batches = np.zeros([n_way, n_query, width, height, channel], dtype=np.float32)
            for i, epi_cls in enumerate(epi_classes):
                selected = np.random.permutation(len(X_dict[epi_cls]))[:n_shot + n_query]
                samples[i] = zgff.getPatches(np.array(X_dict[epi_cls])[selected[:n_shot]])
                batches[i] = zgff.getPatches(np.array(X_dict[epi_cls])[selected[n_shot:]])

            # batches = inner_class_query_augmentation(np.copy(batches), 'rotate')
            samples = samples.transpose((0, 1, 4, 2, 3))
            batches = batches.transpose((0, 1, 4, 2, 3))
            samples = np.reshape(samples, [n_way * n_shot, 1, channel, width, height])
            batches = np.reshape(batches, [n_way * n_query, 1, channel, width, height])
            # calculate features
            sample_features = proto_network(Variable(torch.from_numpy(samples)).cuda(GPU))  # 5x64
            sample_features = sample_features.view(n_way, n_shot, 64*75)
            sample_features = torch.mean(sample_features, 1).squeeze(1)
            test_features = proto_network(Variable(torch.from_numpy(batches)).cuda(GPU))  # 20x64
            test_features = test_features.squeeze()
            target_inds = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()  #biaoqian tensor
            target_inds = Variable(target_inds, requires_grad=False).cuda(GPU)
            dists = euclidean_dist(test_features, sample_features)
            log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)
            loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
            _, y_hat = log_p_y.max(2)
            acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
            proto_network_optim.zero_grad()
            loss_val.backward()
            proto_network_optim.step()
            proto_network_scheduler.step()
            last_epoch_loss_avrg += loss_val.data
            last_epoch_acc_avrg += acc_val.data
            if (epi + 1) % 50 == 0:
                print('[epoch {}/{}, episode {}/{}] => loss: {:.5f}, acc: {:.5f}'.format(ep + 1, n_epochs, epi + 1,
                                                                                         n_episodes, loss_val.data,
                                                                                         acc_val.data))
        if (ep + 1) >= 0:
            proto_network.eval()
            test_num = 3000
            test_count = int(len(yval) / test_num)
            for i in range(test_count):
                query_test = zgff.getPatches(Xval[i * test_num:(i + 1) * test_num]).astype(np.float32)
                query_test = np.reshape(query_test, [-1, 1, width, height, channel])
                query_test = query_test.transpose((0, 1, 4, 2, 3))
                predict_dataset[i * test_num:(i + 1) * test_num] = test(support_test, query_test, proto_network)
            query_test = zgff.getPatches(Xval[test_count * test_num:]).astype(np.float32)
            query_test = np.reshape(query_test, [-1, 1, width, height, channel])
            query_test = query_test.transpose((0, 1, 4, 2, 3))
            predict_dataset[test_count * test_num:] = test(support_test, query_test, proto_network)
            overall_acc = accuracy_score(yval, predict_dataset)
            if overall_acc > best_acc:
                best_acc = overall_acc
                print('best acc: {:.2f}'.format(overall_acc * 100))
                torch.save(proto_network.state_dict(), save_path)
    b = datetime.now()
    durn = (b - a).seconds
    print("Training time:", durn)
    print('Last loss:{:.5f}'.format(last_epoch_loss_avrg / n_episodes))
    print('Last acc:{:.2f}'.format(last_epoch_acc_avrg / n_episodes * 100))
    proto_network.load_state_dict(torch.load(save_path))
    # save_model(proto_network, proto_network_optim, last_epoch + n_epochs, save_path)
    print('completed')
    a = datetime.now()
    print('Testing...')
    proto_network.eval()
    del X_dict
    test_num = 3000
    test_count = int(len(yval) / test_num)
    for i in range(test_count):
        query_test = zgff.getPatches(Xval[i * test_num:(i + 1) * test_num]).astype(np.float32)
        query_test = np.reshape(query_test, [-1, 1, width, height, channel])
        query_test = query_test.transpose((0, 1, 4, 2, 3))
        predict_dataset[i * test_num:(i + 1) * test_num] = test(support_test, query_test, proto_network)
    query_test = zgff.getPatches(Xval[test_count * test_num:]).astype(np.float32)
    query_test = np.reshape(query_test, [-1, 1, width, height, channel])
    query_test = query_test.transpose((0, 1, 4, 2, 3))
    del Xval
    predict_dataset[test_count * test_num:] = test(support_test, query_test, proto_network)
    confusion = confusion_matrix(yval, predict_dataset)
    acc_for_each_class = precision_score(yval, predict_dataset, average=None)
    average_accuracy = np.mean(acc_for_each_class)
    kappa = cohen_kappa_score(yval, predict_dataset)
    overall_acc = accuracy_score(yval, predict_dataset)
    print('OA: {:.2f}'.format(overall_acc * 100))
    print('kappa:{:.4f}'.format(kappa))
    print('PA:')
    for i in range(len(acc_for_each_class)):
        print('{:.2f}'.format(acc_for_each_class[i] * 100))
    print('AA: {:.2f}'.format(average_accuracy * 100))


if __name__ == '__main__':
    main()