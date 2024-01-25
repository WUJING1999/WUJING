import cv2
import numpy as np
from IP_proto_new3d_test_4layer_allband import ProtoNetwork, test
from sklearn.metrics import classification_report, \
    confusion_matrix, \
    cohen_kappa_score, \
    precision_score, \
    accuracy_score
import torch
from common_usages import seed_everything
import matplotlib.pyplot as plt
from classifyUtils import plot_confusion_matrix
from IP_data_loader import IndianPines

GPU = 0
LEARNING_RATE = 0.0005
n_epochs = 60
n_episodes = 100
n_classes = 16
n_way = n_classes
n_shot = 5
n_query = 15
n_test_way = n_classes
n_test_shot = 20


if __name__ == '__main__':
    seed_everything()
    width, height, channel = 11, 11, 200
    zgff = IndianPines(width)
    name = "IP-band200-3layerbaseline"
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
    proto_network = ProtoNetwork(64)
    proto_network.cuda(GPU)
    support_test = np.zeros([n_test_way, n_test_shot, width, height, channel], dtype=np.float32)
    predict_dataset = np.zeros([len(ytest)], dtype=np.int32)
    epi_classes = np.arange(n_classes)
    for i, epi_cls in enumerate(epi_classes):
        selected = np.random.permutation(len(X_dict[epi_cls]))[:n_test_shot]
        support_test[i] = zgff.getPatches(np.array(X_dict[epi_cls])[selected])
    support_test = support_test.transpose((0, 1, 4, 2, 3))
    support_test = np.reshape(support_test, [n_test_way * n_test_shot, 1, channel, width, height])
    save_path = './model/IP/IP_proto_3d-5shotallband-64output.pth'.format(n_way, n_shot)
    proto_network.load_state_dict(torch.load(save_path))
    print('Testing...')
    proto_network.eval()
    del X_dict
    test_num = 1000
    test_count = int(len(ytest) / test_num)
    for i in range(test_count):
        query_test = zgff.getPatches(Xtest[i * test_num:(i + 1) * test_num]).astype(np.float32)
        query_test = np.reshape(query_test, [-1, 1, width, height, channel])
        query_test = query_test.transpose((0, 1, 4, 2, 3))
        predict_dataset[i * test_num:(i + 1) * test_num] = test(support_test, query_test, proto_network)
    query_test = zgff.getPatches(Xtest[test_count * test_num:]).astype(np.float32)
    query_test = np.reshape(query_test, [-1, 1, width, height, channel])
    query_test = query_test.transpose((0, 1, 4, 2, 3))
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

    zgff = IndianPines(height)
    h, w, c = zgff.get_image_size()
    # query_predict = np.zeros([w, 1, height, width, channel], dtype=np.float32)
    predict_image = np.zeros([h, w], dtype=np.float32)
    step = 1
    for i in range(0, h, step):
        indexes = np.arange(i*w, (i + step) * w)
        query_predict = np.expand_dims(zgff.getPatches(indexes), axis=1)
        predict_image[i:i+step] = np.reshape(test(support_test, query_predict.transpose((0, 1, 4, 2, 3)), proto_network), [step, -1])
        print(i)

    R = np.zeros([h, w], dtype='uint8')
    G = np.zeros([h, w], dtype='uint8')
    B = np.zeros([h, w], dtype='uint8')
    for i in range(h):
        for j in range(w):
            R[i, j], G[i, j], B[i, j] = zgff.get_color_by_index(predict_image[i, j])
    cv2.imwrite('./map/{}.png'.format(name), cv2.merge([B, G, R]))
    cv2.imshow('Classifcation Map', cv2.merge([B, G, R]))
    print('complete')
    cv2.waitKey(0)
