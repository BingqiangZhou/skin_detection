import os
import torchvision
import torch
import  numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from dataset import SkinDetectionDataset
from skindetectionnet import SkinDetectionNet
from utils import *

writer = SummaryWriter()

# get our dataset
root_dir = r'.\dataset'
dataset_val = SkinDetectionDataset(root_dir, 'val')

# indices = torch.randperm(len(dataset)).tolist()
# dataset_train = torch.utils.data.Subset(dataset, indices[:5])
# dataset_test = torch.utils.data.Subset(dataset, indices[5:])

# load our model and set directory for saving model parameter
model_pth = r'.\model_pth\final_epoch_20_2019-12-08 02_38_19.840335.pkl'
# model_pth = r'.\model_pth\final_epoch_20_2019-12-08 01_35_29.027475.pkl'

model = SkinDetectionNet()

is_run_in_gpu = True
if is_run_in_gpu == True:
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=model.to(device)

model.eval()

# record start time
begin_time = datetime.now()
print("begin val: ", begin_time)

data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False)
# validate process
for i, data in enumerate(data_loader_val):

    input, ground_truth = data

    if is_run_in_gpu == True:
        input = input.cuda()

    val_image_start_time = datetime.now()
    output = model(input.float())
    val_image_end_time = datetime.now()

    output = output.squeeze(axis=0)

    if is_run_in_gpu == True:
        output = output.cpu().detach()

    # calculate iou
    output[output > 0] = 1
    output[output < 0] = 0

    output = np.squeeze(output.numpy(), axis=0)
    ground_truth = np.squeeze(ground_truth.numpy(), axis=0)
    iou = calculate_iou(output, ground_truth)

    print("image:{}, iou:{}, cost time:{}".format(i, iou, val_image_end_time - val_image_start_time))

    source_image_tag = "image {}/".format(i)
    writer.add_image(source_image_tag, input.squeeze(dim=0))

    ground_truth_tag = "image {}/ground_truth".format(i, iou)
    writer.add_image(ground_truth_tag, ground_truth[np.newaxis,:])

    val_result_tag = "image {}/val result/iou: {}".format(i, iou)
    writer.add_image(val_result_tag, output[np.newaxis,:])

writer.close()

# record end time
end_time = datetime.now()
print("end train: ", end_time)
print("time cost: {}s".format((end_time-begin_time).seconds))