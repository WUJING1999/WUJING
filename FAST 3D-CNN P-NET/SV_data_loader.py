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


class SalinasValley():
    site3_filename = 'data/Salinas_corrected.mat'
    site3_gr_filename = 'data/Salinas_gt.mat'
    site3_gr_classes_filename = 'data/AVIRIS_IndianPine_Site3_classes.csv'

    def __init__(self):

        self.raster_data = sio.loadmat(SalinasValley.site3_filename)['salinas_corrected']
        self.ground_truth = sio.loadmat(SalinasValley.site3_gr_filename)['salinas_gt']
        self.class_df = pd.read_csv(SalinasValley.site3_gr_classes_filename)

    def scale(self):
        scaler = MinMaxScaler()
        raster_array = self.raster_data
        h, w, b = raster_array.shape
        raster_array = raster_array.reshape((b, h, w))
        raster_array = scaler.fit_transform(raster_array.reshape((b, h * w)).T)
        return raster_array.reshape(h, w, b)

    def get_cmap(self):
        colors = [tuple(c) for c in (self.class_df[['R', 'G', 'B']] / 255).values]
        cm = ListedColormap(colors)
        return cm

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

    def get_patch(self, i, j, p=11):
        scaled_data = self.raster_data
        padded_data = np.pad(scaled_data, ((p // 2, p // 2), (p // 2, p // 2), (0, 0)), mode='constant')
        return padded_data[i:i + p, j:j + p, :]

    def get_patches(self, p=11):
        scaled_data = self.raster_data
        padded_data = np.pad(scaled_data, ((p // 2, p // 2), (p // 2, p // 2), (0, 0)), mode='constant')
        patches = image.extract_patches_2d(padded_data, (p, p))
        return patches

    def show_gt(self, ax=None):
        ax = ax or plt.gca()
        gt_data = self.ground_truth

        cm = self.get_cmap()
        ax.imshow(gt_data, cmap=cm)
        ax.set_axis_off()
        bound = list(range(0, 17))

        fig = ax.get_figure()
        fig.set_figwidth(7)
        fig.set_figheight(5)
        plt.legend([mpatches.Patch(color=cm(b)) for b in bound[:-1]],
                   [name for name in list(self.class_df['class_name'])[:-1]],
                   bbox_to_anchor=(1.6, 1),
                   title='Ground Truth\n')
        ax.set_title('Salinas Valley, Ground Truth')

        ax.set_axis_off()
        plt.show()

    def show_pred(self, y_pred, ax=None):
        ax = ax or plt.gca()

        cm = self.get_cmap()
        ax.imshow(y_pred.reshape(512, 217), cmap=cm)
        ax.set_axis_off()
        bound = list(range(0, 17))

        fig = ax.get_figure()
        fig.set_figwidth(7)
        fig.set_figheight(5)
        plt.legend([mpatches.Patch(color=cm(b)) for b in bound],
                   [name for name in list(self.class_df['class_name'])],
                   bbox_to_anchor=(2, 1),
                   title='Ground Truth\n')
        ax.set_title('Salinas Valley, Ground Truth Prediction')

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