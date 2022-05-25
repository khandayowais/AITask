import gzip
import numpy as np
from torch.utils.data import WeightedRandomSampler
import torch
import os

class_shoe = [5, 7, 9]  # shoe classes
IMAGE_SIZE = 28
CHANNEL = 1


def convert_to_binary(labels):
    '''
    Args:
        labels: labels of size N having 10 classes
    return
        labels: labels of size N having 2 classes
    '''
    binary_labels = labels.copy()
    for index in range(len(labels)):
        if labels[index] in class_shoe:
            binary_labels[index] = 1
        else:
            binary_labels[index] = 0

    return binary_labels


def load_mnist(PATH, KIND):
    '''
    Args:
        path: specify the path of the data
        kind: train or test
    return:
        images, labels
    '''
    labels_path = PATH + KIND + '-labels-idx1-ubyte.gz'
    images_path = PATH + KIND + '-images-idx3-ubyte.gz'

    if not os.path.isfile(labels_path):
        print("Files not found ", labels_path)
        exit()
    if not os.path.isfile(images_path):
        print("Files not found ", images_path)
        exit()

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        binary_labels = convert_to_binary(labels)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
        images = np.asarray(images).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, CHANNEL).astype('float32')
        # images = images/255 # Normalizing the data between 0 and 1

    return images, binary_labels


def weighted_sampler(train):
    '''
    Args:
      train: train dataset
    return:
      weighted_sampler: sampeler to draw equal proportion of samples as we have imblanced dataset 18000 from class 1
      and 42000 from class 0
    '''
    targets = train.labels
    class_count = np.unique(targets, return_counts=True)[1]
    weight = 1. / class_count
    samples_weight = weight[targets]
    samples_weight = torch.from_numpy(samples_weight)
    weighted_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    return weighted_sampler



