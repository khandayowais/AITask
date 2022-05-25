from torch.utils.data import Dataset
from utils import load_mnist, weighted_sampler
from os import getcwd
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split


class MNISTDATASET(Dataset):
    '''
    This is the class for the FashionMNIST dataset that can be used by the dataloader
    Args:
        PATH: Path to the data
        kind: train or test
    return:
        Images with their respective labels
    '''

    def __init__(self, PATH, KIND, transform=None):
        self.images, self.labels = load_mnist(PATH, KIND)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def dataset_load(PATH, KIND, BATCH_SIZE):
    '''

    :param PATH: PATH of the dataset
    :param KIND: train or validation or test
    :param BATCH_SIZE: batch size
    :return: train loader or test loader or val_loader
    '''

    # Loading train or test dataset
    # During training we split the test data into test and validation
    
    if KIND == 'validation' or KIND == 'test'or 't10k':
        data = MNISTDATASET(PATH, 't10k', transform=transforms.Compose([transforms.ToTensor()]))
        if KIND == 'validation':
            print('*************** Loading Validation DataLoader ************** ')
            test_size, val_size = get_split_dataset(data, 0.2)
            test_data, val_data = random_split(data, [test_size, val_size])
            val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
            return val_loader
        else:
            print('*************** Loading Test DataLoader ************** ')
            test_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
            return test_loader
    else:
        data = MNISTDATASET(PATH, KIND, transform=transforms.Compose([transforms.ToTensor()]))
        print('*************** Loading Training DataLoader ************** ')
        sampler = weighted_sampler(data)
        train_loader = DataLoader(data, batch_size=BATCH_SIZE, sampler=sampler)
        return train_loader


def get_split_dataset(dataset, split=0.2):
    # test_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=split)
    val_size = int(split * len(dataset))
    test_size = len(dataset) - val_size
    return test_size, val_size
