import os
import numpy as np
import shutil

# Specify the ratio of images that you want to use for training (e.g., 0.8 means 80% of the images go to the training set)
train_ratio = 0.8

# The path to your folder containing the images
folder_path = "data/rawdata/Sononerf_Data_1/images"

# The path to your file containing the poses
poses_path = "data/rawdata/Sononerf_Data_1/poses.txt"

# Read in the poses
with open(poses_path, 'r') as file:
    poses = file.readlines()

# Make sure we have the same number of images and poses
assert len(poses) == len(os.listdir(folder_path))

# Create a permutation of the indices into the images and poses
indices = np.random.permutation(len(poses))

# Split the indices into train and validation indices
split_idx = int(len(poses) * train_ratio)
train_indices, val_indices = indices[:split_idx], indices[split_idx:]

# Write the train and validation poses to separate files
with open('data/poses/train/case0001/train_poses.txt', 'w') as file:
    for idx in train_indices:
        file.write(poses[idx])

with open('data/poses/val/case0001/val_poses.txt', 'w') as file:
    for idx in val_indices:
        file.write(poses[idx])

# Copy the images into separate train and validation directories without renaming
image_filenames = sorted(os.listdir(folder_path))
for i, filename in enumerate(image_filenames):
    old_file_path = os.path.join(folder_path, filename)

    if i in train_indices:
        new_folder_path = "data/images/train/case0001"
    else:
        new_folder_path = "data/images/val/case0001"

    new_file_path = os.path.join(new_folder_path, filename)  # The filename stays the same
    shutil.copy(old_file_path, new_file_path)  # Copy the file instead of moving it
