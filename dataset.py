import os
import torch
import numpy as np
from PIL import Image

class SkinDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        img_dir = os.path.join(self.root_dir, 'train_val')
        img_name = os.listdir(img_dir)
        return len(img_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_dir = os.path.join(self.root_dir, 'train_val')
        ground_truth_dir = os.path.join(self.root_dir, 'temp')

        img_names = sorted(os.listdir(img_dir))
        img_name = img_names[idx]
        img_path = os.path.join(img_dir, img_name)
        image = np.array(Image.open(img_path))

        ground_truth_name = img_name
        ground_truth_path = os.path.join(ground_truth_dir, ground_truth_name)
        ground_truth = np.array(Image.open(ground_truth_path))

        # print("1",image.shape)
        # print('2',ground_truth.shape)
        target = np.zeros(image.shape[:2])
        # target[(ground_truth[:, :] == [253, 231, 36]).all(axis=2)] = 1
        index = (np.array(ground_truth[:, :, 0] > 155, np.int) + np.array(ground_truth[:, :, 2] < 50, np.int) + np.array(
            ground_truth[:, :, 1] < 50, np.int) == 3)
        target[index] = 1

        # print(target.any())

        return image, target