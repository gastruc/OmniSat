import h5py
import numpy as np
import torch
import os
import glob

def replace_nans_with_mean(batch_of_images):
    # Calculate the mean along the specified axis (axis=(2, 3) for temp, channels, height, width)
    image_means = torch.nanmean(batch_of_images, dim=(2, 3), keepdim=True)

    # Create a mask for NaN values
    nan_mask = torch.isnan(batch_of_images)

    # Use PyTorch broadcasting to replace NaN values with corresponding means
    batch_of_images[nan_mask] = image_means.expand_as(batch_of_images)[nan_mask]

    nan_mask = torch.isnan(image_means).any(dim=1).squeeze()

    # Filter out the time steps where there are NaN values
    batch_of_images = batch_of_images[~nan_mask]

    return batch_of_images

data_dir = "././data/TreeSat/" #path of data

with open(data_dir + "train_filenames.lst", 'r') as file:
    lst_list = [line.strip() for line in file.readlines()]

with open(data_dir + "val_filenames.lst", 'r') as file:
    lst_list += [line.strip() for line in file.readlines()]

with open(data_dir + "test_filenames.lst", 'r') as file:
    lst_list += [line.strip() for line in file.readlines()]

for file_name in lst_list:
    if not(os.path.exists(data_dir + "sentinel/" + '.'.join(file_name.split('.')[:-1]) + ".h5")):
        pattern = os.path.join(data_dir, "sentinel", '.'.join(file_name.split('.')[:-1]) + "_*.h5")
        files = glob.glob(pattern)
        
        if len(files) == 1:
            os.rename(files[0], data_dir + "sentinel/" + '.'.join(file_name.split('.')[:-1]) + ".h5")
        else:
            raise ValueError("File not found or multiple files found:", file_name)
            
    with h5py.File(data_dir + "sentinel/" + '.'.join(file_name.split('.')[:-1]) + ".h5", 'r') as file:
        s1_asc = file["sen-1-asc-data"][:]
        s1_asc = replace_nans_with_mean(torch.tensor(s1_asc))
        file_path = data_dir + "s1-asc/" + '.'.join(file_name.split('.')[:-1]) + ".pth"
        s1_asc = torch.cat([s1_asc, (s1_asc[:, 0, :, :]/(s1_asc[:, 1, :, :] + 1e-8)).unsqueeze(1)], dim = 1)
        if torch.isnan(s1_asc).any().item():
            print(file_name)
        torch.save(s1_asc, file_path)
        s1_des = file["sen-1-des-data"][:]
        s1_des = replace_nans_with_mean(torch.tensor(s1_des))
        s1_des = torch.cat([s1_des, (s1_des[:, 0, :, :]/(s1_des[:, 1, :, :] + 1e-8)).unsqueeze(1)], dim = 1)
        if torch.isnan(s1_des).any().item():
            print("s1-des", file_name)
        file_path = data_dir + "s1-des/" + '.'.join(file_name.split('.')[:-1]) + ".pth"
        torch.save(s1_des, file_path)
