import torch
import pickle
import numpy as np
from PIL import Image
import random
from dataset.cifar10 import TransformTwice

class ImdbDatasetForGen(torch.utils.data.Dataset):
    """Imdb dataloader, output image and gender label"""
    
    def __init__(self, lines, transform=None):
        pathlist, labellist = [], []
        for line in lines:
            path, _, label = line.split()
            pathlist.append(path)
            labellist.append(label)
        self.pathlist = pathlist
        self.labellist = labellist
        self.transform = transform

    def __getitem__(self, index):
        path = self.pathlist[index]
        img = Image.open(path).convert("RGB")
        label = int(self.labellist[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.labellist)

class ImdbDatasetwoLabel(torch.utils.data.Dataset):
    """Imdb dataloader, output image and gender label"""
    
    def __init__(self, lines, transform=None):
        pathlist = []
        for line in lines:
            path, _ , _ = line.split()
            pathlist.append(path)
        self.pathlist = pathlist
        self.transform = transform

    def __getitem__(self, index):
        path = self.pathlist[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.pathlist)

def get_imdb(tr_data_path, te_data_path, n_labeled, transform_train, transform_val):
    with open(tr_data_path, "r") as f:
        lines = f.readlines()
    random.shuffle(lines)
    lines_labeled = lines[:n_labeled]
    lines_unlabeled = lines[n_labeled:]
    with open(te_data_path, "r") as f:
        te_lines = f.readlines()
    labeled_ds = ImdbDatasetForGen(lines_labeled, transform=transform_train)
    unlabeled_ds = ImdbDatasetwoLabel(lines_labeled, transform=TransformTwice(transform_train))
    test_ds = ImdbDatasetForGen(te_lines, transform=transform_val)

    print (f"#Labeled: {len(lines_labeled)} #Unlabeled: {len(lines_unlabeled)} #Val: {len(te_lines)}")
    
    return labeled_ds, unlabeled_ds, test_ds, test_ds