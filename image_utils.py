import os

import torch.utils.data
from torchvision import transforms, datasets
from torchvision.utils import save_image

DIR_PATH = os.getcwd() + '/task2/train/'


def load_sampl_data(DIR_PATH, transformer):
    """

    :param DIR_PATH: Image folder path
    :param transf: transformer to be applied
    :return: dataloader
    """
    data = datasets.ImageFolder(DIR_PATH, transform=transformer)
    data_loader = torch.utils.data.DataLoader(data, batch_size=1)

    return data_loader


def save_images(train_loader, count, type):
    for image, label in train_loader:
        dir = os.getcwd() + '/SyntheticImages/train'
        if label == 0:
            dir = dir + '/Flats/'
        else:
            dir = dir + '/Heels/'
        name = type + str(count) + '.png'
        save_image(image, dir + name)


def random_horizontal_flip(count=2):
    for i in range(1, count+1):
        transformer = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor()])
        train_loader = load_sampl_data(DIR_PATH, transformer=transformer)
        save_images(train_loader, i, type='horizontal')
        del transformer, train_loader

def random_vertical_flip(count=2):
    for i in range(1, count+1):
        transformer = transforms.Compose([transforms.RandomVerticalFlip(),transforms.ToTensor()])
        train_loader = load_sampl_data(DIR_PATH, transformer=transformer)
        save_images(train_loader, i, type='vertical')
        del transformer, train_loader
def random_affine(count=2):
    for i in range(1, count+1):
        transf = transforms.Compose([transforms.RandomAffine(i*10), transforms.ToTensor()])
        train_loader = load_sampl_data(DIR_PATH, transformer=transf)
        save_images(train_loader, i, type='affine')
def random_rotation(count=2):
    for i in range(1, count+1):
        transf = transforms.Compose([transforms.RandomRotation(i*2), transforms.ToTensor()])
        train_loader = load_sampl_data(DIR_PATH, transformer=transf)
        save_images(train_loader, i, type='rotate')

def random_rotation_affine(count=2):
    for i in range(1, count+1):
        transf = transforms.Compose([transforms.RandomRotation(i*2), transforms.RandomAffine(i*10),transforms.ToTensor()])
        train_loader = load_sampl_data(DIR_PATH, transformer=transf)
        save_images(train_loader, i, type='rotate_affine')

def gaussian_affine(count=2):
    for i in range(1, count+1,2):
        transf = transforms.Compose([transforms.GaussianBlur(i), transforms.RandomAffine(i*10),transforms.ToTensor()])
        train_loader = load_sampl_data(DIR_PATH, transformer=transf)
        save_images(train_loader, i, type='rotate')

def gaussian_blur(count=2):
    for i in range(1, count+1,2):
        transf = transforms.Compose([transforms.GaussianBlur(i), transforms.ToTensor()])
        train_loader = load_sampl_data(DIR_PATH, transformer=transf)
        save_images(train_loader, i, type='rotate')

def create_synthetic_images():
    count = 100
    #random_horizontal_flip(count=4)
    random_affine(count)
    #random_vertical_flip(count=4)
    #random_rotation(count)
    gaussian_blur(count=28)
    gaussian_affine(count=28)
    random_rotation_affine(count)

create_synthetic_images()