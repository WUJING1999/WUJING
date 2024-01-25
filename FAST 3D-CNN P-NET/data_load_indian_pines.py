from osgeo import gdal
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import image
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io as sio
from IP_data_loader import IndianPines


def savePreprocessedData(X_trainPatches, X_testPatches, y_trainPatches, y_testPatches, windowSize, testRatio = 0.25):
    with open("./data/IPXtrainWindowSize" + str(windowSize) + "testRatio" + str(testRatio) + ".npy", 'bw') as outfile:
        np.save(outfile, X_trainPatches)
    with open("./data/IPXtestWindowSize" + str(windowSize) + "testRatio" + str(testRatio) + ".npy", 'bw') as outfile:
        np.save(outfile, X_testPatches)
    with open("./data/IPytrainWindowSize" + str(windowSize) + "testRatio" + str(testRatio) + ".npy", 'bw') as outfile:
        np.save(outfile, y_trainPatches)
    with open("./data/IPytestWindowSize" + str(windowSize) + "testRatio" + str(testRatio) + ".npy", 'bw') as outfile:
        np.save(outfile, y_testPatches)


def splitTrainTestSet(X, y, testRatio=0.10):
    """
    分割数据集
    :param X:　
    :param y:
    :param testRatio: 分割因子
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=345,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    indianPines = IndianPines(11)
    labels = indianPines.get_gt()
    # data1 = get_patches(data)
    indexes = np.where((labels > 0))
    labels = labels[indexes] - 1
    data1 = indexes[0]
    # print(sequence)
    counts = {}
    for x in labels:
        if x in counts:
            counts[x] += 1
        else:
            counts[x] = 1
    # print(counts)
    for i in sorted(counts):
        print((i, counts[i]), end=" ")

    testRatio = 0.95
    X_train, X_test, Y_train, Y_test = splitTrainTestSet(data1, labels, testRatio)
    # X_train, X_test, y_train, y_test = splitTrainTestSet(indexes[0], labels, 0.95)
    savePreprocessedData(X_train, X_test, Y_train, Y_test, 11, testRatio)
    exit(1)