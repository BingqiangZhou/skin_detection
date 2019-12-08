import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from utils import *


class SkinDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, train_or_val='train', transform=None):
        assert train_or_val in ['train', 'val'], "train_or_val must in ['train', 'val']"

        self.train_or_val = train_or_val
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        img_dir = os.path.join(self.root_dir, self.train_or_val)
        img_name = os.listdir(img_dir)
        return len(img_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_dir = os.path.join(self.root_dir, self.train_or_val)
        ground_truth_dir = os.path.join(self.root_dir, 'groundtruths')

        img_names = sorted(os.listdir(img_dir))
        img_name = img_names[idx]
        img_path = os.path.join(img_dir, img_name)
        image = np.array(Image.open(img_path))

        ground_truth_name = img_name.replace('.jpg','.png')
        ground_truth_path = os.path.join(ground_truth_dir, ground_truth_name)
        ground_truth_image = Image.open(ground_truth_path)
        ground_truth = np.array(ground_truth_image)
        
        ground_truth = transform_image_to_binary_array(ground_truth, (253, 231, 36, 255))

        image = image.transpose((2, 0, 1))

        return image, ground_truth

if __name__ == '__main__':
    root_dir = r'.\dataset'
    dataset_train = SkinDetectionDataset(root_dir,'val')
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True)
    for i, data in enumerate(data_loader_train):
        image, ground_truth = data
        ground_truth = torch.squeeze(ground_truth, dim=0)
        image = torch.squeeze(image, dim=0)
        # plt.imshow(image)
        print(ground_truth[200,100])
        plt.imshow(ground_truth)
        print(image.shape)
        print(ground_truth.shape)
        plt.show()
        break