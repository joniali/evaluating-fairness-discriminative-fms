from torch.utils.data import Dataset

import torch
import numpy as np
import os
from PIL import Image

class MyIdentProf(Dataset):
    """Identity Prof Dataset ."""
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
       
        self.transform = transform
        self.professions = ['chef', 'doctor', 'engineer',\
                            'farmer', 'firefighter', 'judge', \
                             'mechanic', 'police','pilot', 'waiter']        

    def __len__(self):
        # _, _, files = next(os.walk(f"{self.root_dir}/{self.profession}/"))
        # file_count = len(files)
        # return file_count - 1
        return len(self.professions)*900

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        prof_id = int((idx) / 900)
        idx = idx % 900

        images_path = f"{self.root_dir}/{self.professions[prof_id]}/{self.professions[prof_id]}-{idx+1}.jpg"

        # image = io.imread(images_path)
        pil_image = Image.open(images_path).convert("RGB") #(image)

        tran_image = self.transform(pil_image)#.unsqueeze(0)

        return  tran_image, images_path, self.professions[prof_id]