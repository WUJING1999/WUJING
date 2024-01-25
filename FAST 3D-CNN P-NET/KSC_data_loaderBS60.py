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


class KSC():
    site3_filename = 'data/KSC.mat'
    site3_gr_filename = 'data/KSC_gt.mat'
    site3_gr_classes_filename = './palette/KSC_palette.csv'

    def __init__(self, width):
        self.raster_data = sio.loadmat(KSC.site3_filename)['KSC']
        self.image = self.scale()
        self.windowSize = width
        self.margin = int((width - 1) / 2)
        self.ground_truth = sio.loadmat(KSC.site3_gr_filename)['KSC_gt']
        self.class_df = pd.read_csv(KSC.site3_gr_classes_filename)
        self.padded_data = self.padWithZeros()
        self.colorList = self.get_colors()

    def scale(self):
        scaler = MinMaxScaler()
        raster_array = self.raster_data.transpose(2, 0, 1)
        b, h, w = raster_array.shape
        raster_array = scaler.fit_transform(raster_array.reshape((b, h * w)).T)
        return raster_array.reshape(h, w, b)

    def padWithZeros(self):
        newX = np.zeros(
            (self.image.shape[0] + 2 * self.margin, self.image.shape[1] + 2 * self.margin, self.image.shape[2]))
        x_offset = self.margin
        y_offset = self.margin
        newX[x_offset:self.image.shape[0] + x_offset, y_offset:self.image.shape[1] + y_offset, :] = self.image
        return newX

    def get_cmap(self):
        colors = [tuple(c) for c in (self.class_df[['R', 'G', 'B']] / 255).values]
        cm = ListedColormap(colors)
        return cm

    def show(self, patch=None, bands=[29, 19, 9], ax=None):
        ax = ax or plt.gca()
        if patch:
            img = self.get_patch(*patch, 5)[:, :, bands]
            title = 'ksc Site 3 False Color\nPatch ({}, {})'.format(*patch)
        else:
            img = self.scale()[:, :, bands]
            title = 'ksc Site 3\n False Color'

        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(title + '\nBands {}, {}, {}'.format(*bands))

    def get_patch(self, index, p=11):
        i = int(index / self.raster_data.RasterXSize)
        j = index - i * self.raster_data.RasterXSize
        selected =[ 56, 60, 18, 31, 175, 132, 173, 174, 170, 0, 169, 172, 131, 167,
                    171, 41, 163, 73, 2, 27, 168, 96, 133, 75, 165, 166, 161, 159, 164, 162, 42, 160, 139, 34,
                    66, 158, 135, 142, 138, 134, 50, 150, 85, 143, 102, 46, 104, 121, 136, 146, 14, 140, 151, 97, 144, 152, 98, 107,
                    105, 155, ]
        # scaled_data = self.raster_data.ReadAsArray()
        # padded_data = np.pad(scaled_data, ((p // 2, p // 2), (p // 2, p // 2), (0, 0)), mode='constant')
        return self.padded_data[i:i + p, j:j + p, selected]

    def getPatches(self, indexes):
        patchData = np.zeros([len(indexes), self.windowSize, self.windowSize, 60], dtype=np.float32)
        selected = [ 56, 60, 18, 31, 175, 132, 173, 174, 170, 0, 169, 172, 131, 167,
                    171, 41, 163, 73, 2, 27, 168, 96, 133, 75, 165, 166, 161, 159, 164, 162, 42, 160, 139, 34,
                    66, 158, 135, 142, 138, 134, 50, 150, 85, 143, 102, 46, 104, 121, 136, 146, 14, 140, 151, 97, 144, 152, 98, 107,
                    105, 155, ]
        for i, index in enumerate(indexes):
            row = int(index / self.image.shape[1])
            col = int(index - row * self.image.shape[1])
            patchData[i] = self.padded_data[row:row + self.windowSize, col:col + self.windowSize, selected]
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
        ax.set_title('ksc Site 3, Ground Truth')

        ax.set_axis_off()
        plt.show()

    def show_pred(self, y_pred, ax=None):
        ax = ax or plt.gca()

        cm = self.get_cmap()
        ax.imshow(y_pred.reshape(512, 614), cmap=cm)
        ax.set_axis_off()
        bound = list(range(0, 17))

        fig = ax.get_figure()
        fig.set_figwidth(7)
        fig.set_figheight(5)
        plt.legend([mpatches.Patch(color=cm(b)) for b in bound],
                   [name for name in list(self.class_df['class_name'])],
                   bbox_to_anchor=(1.6, 1),
                   title='Legend\n')
        ax.set_title('ksc Site 3, Ground Truth Prediction')

        ax.set_axis_off()
        plt.show()

    def get_gt(self, pixel=None):
        gt_data = self.ground_truth
        if pixel:
            return gt_data[pixel[0], pixel[1]]
        else:
            return gt_data.reshape(gt_data.shape[0] * gt_data.shape[1])

    def get_dataset(self, n=None):
        if n:
            return self.get_patches()[:n], self.get_gt()[:n]
        else:
            return self.get_patches(), self.get_gt()

    def get_colors(self):
        numberList = self.class_df['Number'].values
        colorList = np.zeros([len(numberList), 3], dtype=int)
        for i in numberList:
            row = self.class_df[self.class_df['Number'] == i]
            colorList[i, 0] = row['R'].values[0]
            colorList[i, 1] = row['G'].values[0]
            colorList[i, 2] = row['B'].values[0]
        return colorList

    def get_color_by_index(self, index):
        index = int(index) + 1
        return self.colorList[index, 0], self.colorList[index, 1], self.colorList[index, 2]

    def get_image_size(self):
        return self.image.shape[0], self.image.shape[1], self.image.shape[2]