import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
from fair_pca import apply_fair_PCA_to_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
import numpy as np

def get_features(dataset, model, device):

    all_features = []
    all_labels_categories = []
    # all_labels_captions = []
    all_labels_gender = []

    with torch.no_grad():
        for images, cat, cap, gender in tqdm(DataLoader(dataset, batch_size=100)):
#             print(images.shape)
            features = model.encode_image(images.to(device))
            all_features.append(features)
            # all_labels_categories.append(cat)
            # all_labels_captions.append(cap)
            all_labels_gender.append(gender)

    return torch.cat(all_features), \
            torch.cat(all_labels_gender).cpu().numpy()

def calculate_pca_projection(clip_model, preprocess, device):
    
    import sys
    sys.path.insert(0, '/data/data/fairclip/')
    import customCOCO as cusCoco #import MyCocoDataset

    mycoco_train = cusCoco.MyCocoDataset('/data/data/fairclip/MSCOCO/train2014','/data/data/fairclip/MSCOCO/annotations/instances_train2014.json', '/data/data/fairclip/MSCOCO/annotations/captions_train2014.json', transform = preprocess )
    
    all_features,  all_gender_labels = get_features(mycoco_train, clip_model, device)
    all_features /= all_features.norm(dim=-1, keepdim=True)
    # print("norm shape: ", all_features.norm(dim=-1, keepdim=True).shape)
    # later do inferred from clip model
    idxs = np.where(all_gender_labels!= 2)[0]
    # print("dim fpca: ", all_features.shape, all_features.shape[1]-1)
    pipe_train = apply_fair_PCA_to_dataset((all_features[idxs].cpu().numpy().astype(np.float64), all_gender_labels[idxs], all_gender_labels[idxs]),
        all_features.shape[1]-1, 
        LogisticRegression,
        'selection_rate_parity', 0, 
        standardize=False, fit_classifier=False)
    return pipe_train


def main_fpca(clip_model_type: str):
    device = torch.device('cuda:1')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./data/coco/oscar_split_{clip_model_name}_fpca_train.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    projection_pca = calculate_pca_projection(clip_model, preprocess, device)
    with open('./data/coco/train_caption.json', 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(data))):
        d = data[i]
        img_id = d["image_id"]
        filename = f"../MSCOCO/train2014/COCO_train2014_{int(img_id):012d}.jpg"
        if not os.path.isfile(filename):
            filename = f"../MSCOCO/val2014/COCO_val2014_{int(img_id):012d}.jpg"
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
            # print("prefix before: ", prefix.shape)
            # print(prefix.shape,  prefix.norm(dim=-1, keepdim=True).shape)
            prefix /= prefix.norm(dim=-1, keepdim=True)
            # print(prefix.shape, prefix.norm(dim=-1, keepdim=True))
            prefix = projection_pca.just_transform(prefix.cpu().numpy().astype(np.float64))
            prefix = torch.tensor(prefix).to(torch.float16)
            # print("prefix after: ", prefix.shape)
            # input(".......")
        # print(" ********* clip_embedding:  ", i, prefix.shape)
        d["clip_embedding"] = i
        all_embeddings.append(prefix)
        all_captions.append(d)
        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0

def main_normal(clip_model_type: str):
    device = torch.device('cuda:2')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./data/coco/oscar_split_{clip_model_name}_normal_train.pkl"
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    
    with open('./data/coco/train_caption.json', 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(data))):
        d = data[i]
        img_id = d["image_id"]
        filename = f"../MSCOCO/train2014/COCO_train2014_{int(img_id):012d}.jpg"
        if not os.path.isfile(filename):
            filename = f"../MSCOCO/val2014/COCO_val2014_{int(img_id):012d}.jpg"
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
            # print("prefix before: ", prefix.shape)
            # print(prefix.shape,  prefix.norm(dim=-1, keepdim=True).shape)
            prefix /= prefix.norm(dim=-1, keepdim=True)
            # print(prefix.shape, prefix.norm(dim=-1, keepdim=True))
            
            # print("prefix after: ", prefix.shape)
            # input(".......")
        # print(" ********* clip_embedding:  ", i, prefix.shape)
        d["clip_embedding"] = i
        all_embeddings.append(prefix)
        all_captions.append(d)
        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0

def calc_mutual_info(clip_model, preprocess, device):

    import sys
    sys.path.insert(1, '/data/data/fairclip/Mitigate-Gender-Bias-in-Image-Search')
    import utils as ut_mitigate
    sys.path.insert(0, '/data/data/fairclip/')
    import customCOCO as cusCoco #import MyCocoDataset

    mycoco_train = cusCoco.MyCocoDataset('/data/data/fairclip/MSCOCO/train2014','/data/data/fairclip/MSCOCO/annotations/instances_train2014.json', '/data/data/fairclip/MSCOCO/annotations/captions_train2014.json', transform = preprocess )

    all_features,  all_gender_labels = get_features(mycoco_train, clip_model, device)
    all_features /= all_features.norm(dim=-1, keepdim=True)
    mis = []
    for col in range(all_features.shape[1]):
        mi = ut_mitigate.mutual_information_2d(all_features[:,col].squeeze().cpu().numpy(), all_gender_labels)
        mis.append((mi, col))
    mis = sorted(mis, reverse=False)
    mis = np.array([l[1] for l in mis])

    return mis    

def main_clip_clip400(clip_model_type: str):

    device = torch.device('cuda:3')
    clip_model_name = clip_model_type.replace('/', '_')
    out_path_400 = f"./data/coco/oscar_split_{clip_model_name}_train_clip400.pkl"
    out_path_256 = f"./data/coco/oscar_split_{clip_model_name}_train_clip256.pkl"

    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    mis = calc_mutual_info(clip_model, preprocess, device)
    # num_clip = 400
    with open('./data/coco/train_caption.json', 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    all_embeddings_400 = []
    all_captions_400 = []
    all_embeddings_256 = []
    all_captions_256 = []
    for i in tqdm(range(len(data))):
        d = data[i]
        img_id = d["image_id"]
        filename = f"../MSCOCO/train2014/COCO_train2014_{int(img_id):012d}.jpg"
        if not os.path.isfile(filename):
            filename = f"../MSCOCO/val2014/COCO_val2014_{int(img_id):012d}.jpg"
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()
            # print("prefix before: ", prefix.shape)
            # print(prefix.shape,  prefix.norm(dim=-1, keepdim=True).shape)
            prefix /= prefix.norm(dim=-1, keepdim=True)
            prefix_400 = prefix[:, mis[:400]]
            prefix_256 = prefix[:, mis[:256]]

            # print(prefix.shape, prefix.norm(dim=-1, keepdim=True))
            # prefix = projection_pca.just_transform(prefix.cpu().numpy().astype(np.float64))
            # prefix = torch.tensor(prefix).to(torch.float16)
            # print("prefix after: ", prefix_400.shape, prefix_256.shape)
            # input(".......")
        # 
        d["clip_embedding"] = i
        # print(" ********* clip_embedding:  ", d)#, prefix.shape)
        all_embeddings_400.append(prefix_400)
        all_embeddings_256.append(prefix_256)
        all_captions_400.append(d)
        if (i + 1) % 10000 == 0:
            with open(out_path_400, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings_400, dim=0), "captions": all_captions_400}, f)
            with open(out_path_256, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings_256, dim=0), "captions": all_captions_400}, f)

    with open(out_path_400, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings_400, dim=0), "captions": all_captions_400}, f)

    with open(out_path_256, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings_256, dim=0), "captions": all_captions_400}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings_400))
    print("%0d embeddings saved " % len(all_embeddings_256))

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-B/32", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    args = parser.parse_args()
    if True:
        print("Computing embeddings with FPCA")
        exit(main_fpca(args.clip_model_type))
    if False:
        print("Computing embeddings with CLIP CLIP")
        exit(main_clip_clip400(args.clip_model_type))
    if False:
        print("Computing embeddings normal")
        exit(main_normal(args.clip_model_type))
