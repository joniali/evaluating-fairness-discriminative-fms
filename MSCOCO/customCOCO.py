from torch.utils.data import Dataset
import torchvision.datasets as dset
import torch
import numpy as np
class MyCocoDataset(Dataset):
    """Coco Dataset ."""

    def __init__(self, root_dir, ann_file, cap_file, transform=None, img = True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img = img
        self.coco_ann = dset.CocoDetection(root = root_dir,
                                annFile = ann_file,transform = transform)
        self.coco_cap = dset.CocoCaptions(root = root_dir,
                                annFile = cap_file,transform = transform)
        
            

    def __len__(self):
        return len(self.coco_ann)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image , ann = self.coco_ann[idx]
        image , cap = self.coco_cap[idx]
        if len(cap) > 5:
            cap = cap[:5]
        cat_ids = [ann_['category_id'] for ann_ in ann]
        cat_array = np.zeros(91)
        gender = -1 
        # print("cat_ids:", cat_ids)
        # print(np.where(np.asarray(cat_ids) == 1)[0].shape[0])
        if np.where(np.asarray(cat_ids) == 1)[0].shape[0] == 1:
            male_words = ['man', 'men', 'boy','boys', 'male','males', 'gentleman', 'gentlemen']
            female_words = ['woman', 'women', 'girl', 'girls', 'female', 'females', 'lady', 'ladies']
            for c in cap:
                c = c.lower().replace("\n", "")
                if gender == 2:
                    break
                ml = [m in c.split(" ") for m in male_words]
    #             print("male: ", ml)
                if any(ml):
                    if gender == 0:
                        gender = 2
                    else:
                        gender = 1
                fe_ml = [m in c.split(" ") for m in female_words]
    #             print("female: ", fe_ml)
                if any(fe_ml):
                    if gender == 1:
                        gender = 2
                    else:
                        gender = 0
        if gender == -1:
            gender = 2
        cat_array[cat_ids] = 1

        if self.img:
            return image, cat_array, cap, gender
        else:
            return cat_array, cap, gender


