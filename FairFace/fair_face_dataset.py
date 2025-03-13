import os
import torch
import warnings
import numpy as np
import pandas as pd
from PIL import Image
# import matplotlib.pyplot as plt
# plt.ion()
from torch.utils.data import Dataset, DataLoader
# from __future__ import print_function, division
warnings.filterwarnings("ignore")


class FairFaceDataset(Dataset):
    """FairFace Dataset ."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.fairface_image_data = pd.read_csv(csv_file)
        self.transform = transform
        self.attribute_to_integer_dict = {}
        self.attribute_to_integer_dict_inverse = {}
        self.attribute_count_dict = {}
        for attribute_name in ['age', 'gender', 'race']:
            self.attribute_to_integer_dict[attribute_name] = {}
            self.attribute_to_integer_dict_inverse[attribute_name] = {}
            for counter, attr in enumerate(sorted(pd.unique(self.fairface_image_data[attribute_name]).tolist())):
                self.attribute_to_integer_dict[attribute_name][attr] = counter
                self.attribute_to_integer_dict_inverse[attribute_name][counter] = attr
            self.attribute_count_dict[attribute_name] = counter + 1
            self.fairface_image_data[attribute_name] = self.fairface_image_data[attribute_name].map(self.attribute_to_integer_dict[attribute_name])
            
    def __len__(self):
        return len(self.fairface_image_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name =  os.path.join(self.root_dir,self.fairface_image_data.iloc[idx,0])
        
        image = Image.open(img_name).convert("RGB") #io.imread(img_name)
        
        race = self.fairface_image_data.iloc[idx,3]

        if self.transform:
            image = self.transform(image)

        label_age = self.fairface_image_data.iloc[idx, 1]
        label_gender = self.fairface_image_data.iloc[idx, 2]
        label_race = self.fairface_image_data.iloc[idx, 3]
        return image, label_age, label_gender, label_race

