from torch.utils.data import Dataset
import numpy as np
import json
from skmultilearn.model_selection import iterative_train_test_split
import torch
import rasterio
from datetime import datetime
import os
from pathlib import Path 
from random import shuffle
import pandas as pd

def collate_fn(batch):
    """
    Collate function for the dataloader.
    Args:
        batch (list): list of dictionaries with keys "label", "name"  and the other corresponding to the modalities used
    Returns:
        dict: dictionary with keys "label", "name"  and the other corresponding to the modalities used
    """
    keys = list(batch[0].keys())
    output = {}
    for key in ["s2", "s1-asc", "s1-des", "s1"]:
        if key in keys:
            idx = [x[key] for x in batch]
            max_size_0 = max(tensor.size(0) for tensor in idx)
            stacked_tensor = torch.stack([
                    torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, 0, 0, max_size_0 - tensor.size(0)))
                    for tensor in idx
                ], dim=0)
            output[key] = stacked_tensor
            keys.remove(key)
            key = '_'.join([key, "dates"])
            idx = [x[key] for x in batch]
            max_size_0 = max(tensor.size(0) for tensor in idx)
            stacked_tensor = torch.stack([
                    torch.nn.functional.pad(tensor, (0, max_size_0 - tensor.size(0)))
                    for tensor in idx
                ], dim=0)
            output[key] = stacked_tensor
            keys.remove(key)
    if 'name' in keys:
        output['name'] = [x['name'] for x in batch]
        keys.remove('name')
    for key in keys:
        output[key] = torch.stack([x[key] for x in batch])
    return output

def read_dates(txt_file: str) -> np.array:
    with open(txt_file, 'r') as f:
        products= f.read().splitlines()
    dates = []
    for file in products:
        date_object = datetime.strptime(file[11:19], '%Y%m%d')
        dates.append(date_object.timetuple().tm_yday)
    return torch.tensor(dates)

class FLAIR(Dataset):
    def __init__(
        self,
        path,
        modalities,
        transform,
        split: str = "train",
        partition: float = 1.,
        num_classes: int = 13,
        crop_s2: bool = True,
        sat_patch_size: int = 40,
        ):
        """
        Initializes the dataset.
        Args:
            path (str): path to the dataset
            modalities (list): list of modalities to use
            transform (torchvision.transforms): transform to apply to the data
            split (str): split to use (train, val, test)
            partition (float): proportion of the dataset to keep
            num_classes (int): number of classes
        """
        self.path = path
        self.transform = transform
        self.partition = partition
        self.modalities = modalities
        self.num_classes = num_classes
        self.crop_s2 = crop_s2
        self.sat_patch_size = sat_patch_size
        self.data, self.labels = self.load_data(split)
        self.collate_fn = collate_fn

    def load_data (self, split, val_percent=0.8): 
        """ Returns dicts (train/val/test) with 6 keys: 
        - PATH_IMG : aerial image (path, str) 
        - PATH_SP_DATA : satellite image (path, str) 
        - PATH_SP_DATES : satellite product names (path, str) 
        - PATH_SP_MASKS : satellite clouds / snow masks (path, str)
        - SP_COORDS : centroid coordinate of patch in superpatch (list, e.g., [56,85]) 
        - PATH_LABELS : labels (path, str)
        """ 
        def get_data_paths(path_domains, path_data, matching_dict, test_set): 
            #### return data paths 
            def list_items(path, filter): 
                for path in Path(path).rglob(filter): 
                    yield path.resolve().as_posix() 
            status = ['train' if test_set == False else 'test'][0] 
            ## data paths dict
            if status == 'test':
                paths_data = {'_'.join(["path", m]): path_data + '_'.join(["/flair_2", m, status]) for m in ["aerial", "sen", "labels"]}
            else:
                paths_data = {'_'.join(["path", m]): path_data + '_'.join(["/flair", m, status]) for m in ["aerial", "sen", "labels"]}
            data = {'PATH_IMG':[], 'PATH_SP_DATA':[], 'SP_COORDS':[], 'PATH_SP_DATES':[],  'PATH_SP_MASKS':[], 'PATH_LABELS':[]}
            for domain in path_domains: 
                for area in os.listdir(Path(paths_data['path_aerial'], domain)): 
                    aerial = sorted(list(list_items(Path(paths_data['path_aerial'])/domain/Path(area), 'IMG*.tif')), key=lambda x: int(x.split('_')[-1][:-4])) 
                    sen2sp = sorted(list(list_items(Path(paths_data['path_sen'])/domain/Path(area), '*data.npy'))) 
                    sprods = sorted(list(list_items(Path(paths_data['path_sen'])/domain/Path(area), '*products.txt')))
                    smasks = sorted(list(list_items(Path(paths_data['path_sen'])/domain/Path(area), '*masks.npy')))
                    coords = [] 
                    for k in aerial: 
                        coords.append(matching_dict[k.split('/')[-1]]) 
                    data['PATH_IMG'] += aerial 
                    data['PATH_SP_DATA'] += sen2sp*len(aerial) 
                    data['PATH_SP_DATES'] += sprods*len(aerial)
                    data['PATH_SP_MASKS'] += smasks*len(aerial) 
                    data['SP_COORDS'] += coords
                    data['PATH_LABELS'] += sorted(list(list_items(Path(paths_data['path_labels'])/domain/Path(area), 'MSK*.tif')), key=lambda x: int(x.split('_')[-1][:-4])) 

            labels = np.zeros((len(data['PATH_LABELS']), self.num_classes))
            print("Converting to multilabel task")
            for i, path in enumerate(data['PATH_LABELS']):
                with rasterio.open(path) as f:
                    labs = np.unique(f.read()[0])
                    labs[labs > self.num_classes] = self.num_classes
                    labs = labs-1
                    labels[i, labs] = 1
            del data['PATH_LABELS']
            lines, labels, _, _ = iterative_train_test_split(np.expand_dims(np.arange(0, len(labels)), axis=1), labels, test_size = 1. - self.partition)
            data = pd.DataFrame(data).iloc[np.squeeze(lines)]
            return data, labels
                    
        with open(self.path + "/flair-2_centroids_sp_to_patch.json", 'r') as file: 
            matching_dict = json.load(file)

        if split == 'test':
            path = Path(self.path + "flair_2_aerial_test")
            domain = os.listdir(path)
        else:
            path = Path(self.path + "flair_aerial_train")
            trainval_domains = os.listdir(path)
            shuffle(trainval_domains) 
            idx_split = int(len(trainval_domains) * val_percent) 
            if split == 'train':
                domain = trainval_domains[:idx_split]
            else:
                domain = trainval_domains[idx_split:]
        df, labels = get_data_paths(domain, self.path, matching_dict, test_set=(split == 'test'))     
        return df, labels
    
    def read_superarea_and_crop(self, numpy_file: str, idx_centroid: list) -> np.ndarray:
        data = np.load(numpy_file, mmap_mode='r')
        subset_sp = data[:,:,idx_centroid[0]-int(self.sat_patch_size/2):idx_centroid[0]+int(self.sat_patch_size/2),idx_centroid[1]-int(self.sat_patch_size/2):idx_centroid[1]+int(self.sat_patch_size/2)]
        return subset_sp

    def __getitem__(self, i):
        """
        Returns an item from the dataset.
        Args:
            i (int): index of the item
        Returns:
            dict: dictionary with keys "label", "name" and the other corresponding to the modalities used
        """
        line = self.data.iloc[i]
        output = {'label': torch.tensor(self.labels[i]), 'name': line['PATH_IMG']}

        if 'aerial' in self.modalities:
            with rasterio.open(line['PATH_IMG']) as f:
                output["aerial"] = torch.FloatTensor(f.read())

        if 's2' in self.modalities:
            X, Y = line['SP_COORDS']
            if self.crop_s2:
                output["s2"]= torch.FloatTensor(np.load(line['PATH_SP_DATA'])[:,:,X - 5:X + 5, Y - 5:Y + 5].astype(float))
            else:
                output["s2"]= torch.FloatTensor(self.read_superarea_and_crop(line['PATH_SP_DATA'], line["SP_COORDS"]).astype(float))
            output["s2_dates"] = read_dates(line['PATH_SP_DATES'])
            N = len(output["s2_dates"])
            if N > 50:
                random_indices = torch.randperm(N)[:50]
                output["s2"] = output["s2"][random_indices]
                output["s2_dates"] = output["s2_dates"][random_indices]
        
        if "s2-4season-median" in self.modalities:
            X, Y = line['SP_COORDS']
            output_inter= torch.FloatTensor(np.load(line['PATH_SP_DATA'])[:,:,X - 5:X + 5, Y - 5:Y + 5].astype(float))
            dates = read_dates(line['PATH_SP_DATES'])
            l = []
            for i in range (4):
                mask = ((dates >= 92 * i) & (dates < 92 * (i+1)))
                if sum(mask) > 0:
                    r, _ = torch.median(output_inter[mask], dim = 0)
                    l.append(r)
                else:
                    l.append(torch.zeros((output_inter.shape[1], output_inter.shape[-2], output_inter.shape[-1])))
            output["s2-4season-median"] = torch.cat(l)

        if "s2-median" in self.modalities:
            X, Y = line['SP_COORDS']
            output["s2-median"] , _ = torch.median(torch.FloatTensor(np.load(line['PATH_SP_DATA'])[:,:,X - 5:X + 5, Y - 5:Y + 5].astype(float)), dim = 0)
       
        return self.transform(output)

    def __len__(self):
        return len(self.data)
