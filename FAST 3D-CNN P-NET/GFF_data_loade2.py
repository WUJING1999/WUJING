from osgeo import gdal
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_extraction import image
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io as sio
from sklearn import preprocessing
from sklearn.decomposition import PCA


class GFFTreeSpecies():
    site3_filename = './data/sg_quac_test2.tif'
    site3_gr_filename = './data/sg_quac_test2_gt.tif'
    site3_gr_classes_filename = './palette/ZGFF_palette.csv'

    def __init__(self, width):
        self.raster_data = gdal.Open(GFFTreeSpecies.site3_filename)
        # img = self.raster_data.ReadAsArray()
        # img = img.transpose((1, 2, 0))
        self.image = self.scale()
        # self.image = self.applyPCA(numComponents=40)
        self.ground_truth = gdal.Open(GFFTreeSpecies.site3_gr_filename)
        self.class_df = pd.read_csv(GFFTreeSpecies.site3_gr_classes_filename)
        margin = int((width - 1) / 2)
        self.padded_data = self.padWithZeros(margin)
        self.colorList = self.get_colors()

    def applyPCA(self, numComponents=75):
        newX = np.reshape(self.image, (-1, self.image.shape[2]))
        pca = PCA(n_components=numComponents, whiten=True)
        newX = pca.fit_transform(newX)
        newX = np.reshape(newX, (self.image.shape[0], self.image.shape[1], numComponents))
        return newX

    def scale(self):
        scaler = StandardScaler()
        raster_array = self.raster_data.ReadAsArray()
        b, h, w = raster_array.shape
        raster_array = scaler.fit_transform(raster_array.reshape((b, h * w)).T)
        return raster_array.reshape(h, w, b)

    def get_cmap(self):
        colors = [tuple(c) for c in (self.class_df[['R', 'G', 'B']] / 255).values]
        cm = ListedColormap(colors)
        return cm


    def padWithZeros(self, margin=2):
        newX = np.zeros((self.image.shape[0] + 2 * margin, self.image.shape[1] + 2 * margin, self.image.shape[2]))
        x_offset = margin
        y_offset = margin
        newX[x_offset:self.image.shape[0] + x_offset, y_offset:self.image.shape[1] + y_offset, :] = np.copy(self.image)
        return newX

    def show(self, patch=None, bands=[29, 19, 9], ax=None):
        ax = ax or plt.gca()
        if patch:
            img = self.get_patch(*patch, 5)[:, :, bands]
            title = 'Indian Pines Site 3 False Color\nPatch ({}, {})'.format(*patch)
        else:
            img = self.scale()[:, :, bands]
            title = 'Indian Pines Site 3\n False Color'

        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(title + '\nBands {}, {}, {}'.format(*bands))

    def standartizeData(self, X):
        newX = np.reshape(X, (-1, X.shape[2]))
        scaler = preprocessing.StandardScaler().fit(newX)
        newX = scaler.transform(newX)
        newX = np.reshape(newX, (X.shape[0], X.shape[1], X.shape[2]))
        return newX, scaler

    def get_patch(self, index, p=11):
        i = int(index/self.raster_data.RasterXSize)
        j = index - i * self.raster_data.RasterXSize
        # scaled_data = self.raster_data.ReadAsArray()
        # padded_data = np.pad(scaled_data, ((p // 2, p // 2), (p // 2, p // 2), (0, 0)), mode='constant')
        return self.padded_data[i:i + p, j:j + p, :]

    def getPatches(self, indexes, windowSize=11):
        patchData = np.zeros([len(indexes), windowSize, windowSize, self.image.shape[2]], dtype=np.float32)
        for i, index in enumerate(indexes):
            row = int(index / self.image.shape[1])
            col = int(index - row * self.image.shape[1])
            patchData[i] = self.padded_data[row:row + windowSize, col:col + windowSize, :]
        return patchData

    def show_gt(self, ax=None):
        ax = ax or plt.gca()
        gt_data = self.ground_truth.ReadAsArray()

        cm = self.get_cmap()
        ax.imshow(gt_data, cmap=cm)
        ax.set_axis_off()
        bound = list(range(0, 17))

        fig = ax.get_figure()
        fig.set_figwidth(7)
        fig.set_figheight(5)
        plt.legend([mpatches.Patch(color=cm(b)) for b in bound],
                   [name for name in list(self.class_df['class_name'])],
                   bbox_to_anchor=(1.6, 1),
                   title='Ground Truth\n')
        ax.set_title('Indian Pines Site 3, Ground Truth')

        ax.set_axis_off()
        plt.show()

    def show_pred(self, y_pred, ax=None):
        ax = ax or plt.gca()

        cm = self.get_cmap()
        ax.imshow(y_pred.reshape(145, 145), cmap=cm)
        ax.set_axis_off()
        bound = list(range(0, 17))

        fig = ax.get_figure()
        fig.set_figwidth(7)
        fig.set_figheight(5)
        plt.legend([mpatches.Patch(color=cm(b)) for b in bound],
                   [name for name in list(self.class_df['class_name'])],
                   bbox_to_anchor=(1.6, 1),
                   title='Legend\n')
        ax.set_title('Indian Pines Site 3, Ground Truth Prediction')

        ax.set_axis_off()
        plt.show()

    def get_gt(self, pixel=None):
        gt_data = self.ground_truth.ReadAsArray()
        if pixel:
            return gt_data[pixel[0], pixel[1]]
        else:
            return gt_data.reshape(gt_data.shape[0] * gt_data.shape[1])

    def get_dataset(self, n=None):
        if n:
            return self.get_patches()[:n], self.get_gt()[:n]
        else:
            return self.get_patches(), self.get_gt()

    def get_image(self):
        return self.padded_data

    def get_image_size(self):
        return self.image.shape[0], self.image.shape[1], self.image.shape[2]

    def get_colors(self):
        numberList = self.class_df['Number'].values
        colorList = np.zeros([len(numberList), 3], dtype=np.int)
        for i in numberList:
            row = self.class_df[self.class_df['Number'] == i]
            colorList[i, 0] = row['R'].values[0]
            colorList[i, 1] = row['G'].values[0]
            colorList[i, 2] = row['B'].values[0]

        return colorList

    def get_color_by_index(self, index):
        index = int(index) + 1
        return self.colorList[index, 0], self.colorList[index, 1], self.colorList[index, 2]
