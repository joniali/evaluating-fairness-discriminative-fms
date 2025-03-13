from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
# from skimage import io
from PIL import Image
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()

class MIAPDataset(Dataset):
    """MIAP dataset."""

    def __init__(self,  root_dir, split, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        
        self.miap_image_names = pd.read_csv(f'../../miap/open_images_extended_miap_images_{split}.lst', header = None)
        self.annotations =  pd.read_csv(f'../../miap/open_images_extended_miap_boxes_{split}.csv')
        self.transform = transform
        self.gender_dict = {'Predominantly Feminine':0,'Predominantly Masculine':1,'Unknown':2}
        self.age_dict = {'Young':0,'Middle':1,'Older':1,'Unknown':3}

        
    def __len__(self):
        return len(self.miap_image_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name =  os.path.join(self.root_dir,self.miap_image_names.iloc[idx,0].split("/")[1])
        data_ = self.annotations[self.annotations['ImageID']==self.miap_image_names.iloc[idx,0].split("/")[1]]
        
#         print( )
        image = Image.open(img_name+ '.jpg')#.convert("RGB") #io.imread(img_name)
        
        x_min, y_min, x_max,y_max,gens,ag = data_['XMin'],data_['YMin'],data_['XMax'],data_['YMax'],data_['GenderPresentation'],data_['AgePresentation']
        
        if self.transform:
            image = self.transform(image)
        areas = []
        genders = []
        ages = []
        
        for (x_m,y_m,x_ma,y_ma, g,a) in zip(x_min, y_min, x_max,y_max,gens,ag):  
            areas.append((x_ma - x_m) * (y_ma - y_m))
            genders.append(self.gender_dict[g])
            ages.append(self.age_dict[a])
            
        if all(g == 0 for g in genders):
            gen_ret = 0
        elif all(g == 1 for g in genders):
            gen_ret = 1
        else:
            gen_ret = 2
        
        if all(g == 0 for g in ages):
            age_ret = 0
        elif all(g == 1 for g in ages):
            age_ret = 1
        else:
            age_ret = 2
          
        area_ret = 0
        if any(g > 0.35 for g in areas):
            area_ret = 1
        
        ret_no_people = 0
        if len(areas) > 1:
            ret_no_people = 1

        return image,  gen_ret, age_ret, area_ret, ret_no_people