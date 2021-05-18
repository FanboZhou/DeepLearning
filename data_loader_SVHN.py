import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from scipy.io import loadmat
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Run this cell to load the dataset

train = loadmat('/kaggle/input/svhndataset/train_32x32.mat')
test = loadmat('/kaggle/input/svhndataset/test_32x32.mat')

train_img = np.array(train['X'])
test_img = np.array(test['X'])

train_label = train['y']
test_label = test['y']

train_img = np.moveaxis(train_img, -1, 0)
test_img = np.moveaxis(test_img, -1, 0)

def plot_images(img, labels, nrows, ncols):
    """ Plot nrows x ncols images
    """
    fig, axes = plt.subplots(nrows, ncols)
    for i, ax in enumerate(axes.flat): 
        if img[i].shape == (32, 32, 3):
            ax.imshow(img[i])
        else:
            ax.imshow(img[i,:,:,0])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(labels[i])

train_label[train_label == 10] = 0
test_label[test_label == 10] = 0
def rgb2gray(images):
    return np.expand_dims(np.dot(images, [0.2990, 0.5870, 0.1140]), axis=3)

X_train = rgb2gray(train_img).astype(np.float32)
X_test = rgb2gray(test_img).astype(np.float32)

X_train = X_train/255.0
X_test = X_test/255.0

from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
y_train = lb.fit_transform(train_label)
y_test = lb.fit_transform(test_label)