import random
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import os


def gen_train_test(frac_train, num, seed=0, is_symmetric_input=False):
    # Generate train and test split
    if is_symmetric_input:
        pairs = [(i, j) for i in range(num) for j in range(num) if i <= j]
    else:
        pairs = [(i, j) for i in range(num) for j in range(num)]
    random.seed(seed)
    random.shuffle(pairs)
    div = int(frac_train * len(pairs))
    return pairs[:div], pairs[div:]


def train_test_split(p, train, test):
    is_train = []
    is_test = []
    for x in range(p):
        for y in range(p):
            if (x, y) in train:
                is_train.append(True)
                is_test.append(False)
            else:
                is_train.append(False)
                is_test.append(True)
    is_train = np.array(is_train)
    is_test = np.array(is_test)
    return is_train, is_test


class ArithmeticDataset(torch.utils.data.Dataset):
    def __init__(self, data, fn):
        self.fn = fn
        self.data = torch.tensor(data)
        self.labels = torch.tensor([fn(i, j) for i, j in self.data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data, label


class ArithmeticDataModule:
    def __init__(self, train, test, fn, batch_size=1):
        self.fn = fn
        self.train_dataset = ArithmeticDataset(train, fn)
        self.test_dataset = ArithmeticDataset(test, fn)
        self.batch_size = batch_size

    def get_dataloader(self):
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )
        return train_dataloader, test_dataloader


class MNISTDataModule:
    def __init__(self, num_batch):
        self.transform = transforms.Compose([transforms.ToTensor()])

        os.makedirs("./data", exist_ok=True)

        self.train_dataset = datasets.MNIST(
            "./data", train=True, download=True, transform=self.transform
        )

        self.test_dataset = datasets.MNIST(
            "./data", train=False, transform=self.transform
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=num_batch, shuffle=False
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=10000, shuffle=False
        )

    def get_dataloader(self):
        return self.train_dataloader, self.test_dataloader
