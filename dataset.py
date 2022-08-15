import numpy as np
import idx2numpy
import torch
from torch.utils.data import Dataset


class MNIST:
    def __init__(self, image_file, image_label, train=True):
        self.image_file = idx2numpy.convert_from_file(image_file)
        self.image_label = idx2numpy.convert_from_file(image_label)
        self.train = train
        if self.train:
            self.length = 60000
        else:
            self.length = 10000

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        assert idx < self.length, "Index out of bonds"
        return self.image_file[idx], self.image_label[idx]


class MNISTBags(Dataset):
    def __init__(self, bags, label):
        self.bags = bags
        self.label = label

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        image = self.bags[index]
        data_vector = np.array([(image[x].ravel()) / 255.0 for x in range(len(image))])
        data_vector = torch.Tensor(data_vector)
        label = torch.from_numpy(self.label[index])

        return data_vector, label
