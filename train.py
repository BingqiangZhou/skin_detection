import os
import torchvision
import torch
import  numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from dataset import SkinDetectionDataset
from skindetectionnet import SkinDetectionNet

writer = SummaryWriter()

# get our dataset
root_dir = r'.\dataset'
dataset_train = SkinDetectionDataset(root_dir, 'train')

# indices = torch.randperm(len(dataset)).tolist()
# dataset_train = torch.utils.data.Subset(dataset, indices[:5])
# dataset_test = torch.utils.data.Subset(dataset, indices[5:])

# load our model and set directory for saving model parameter
model_pth = r'.\model_pth'

model = SkinDetectionNet()
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=model.to(device)
model.eval()

# set loss function and optimizer
criterion = torch.nn.BCEWithLogitsLoss()
# criterion = torch.nn.L1Loss()
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# set epochs and how many epochs to save the model and output image
num_epochs = 100
num_epochs_save = 10

# when iou > 'accuracy_exceed_precent', end train.
accuracy_exceed_precent = 0.9
accuracy_exceed_precent_flag = False

# set whether to end training, when  iou > 'accuracy_exceed_precent'
is_allow_end_train = False

# record current intersection_over_union(iou)
iou = 0

# record start time
begin_time = datetime.now()
print("begin train: ", begin_time)

# create related directory to save file
pth_dir = os.path.join(model_pth,str(begin_time).replace(':','-')) 
os.mkdir(pth_dir)

# Starting train and validate.
for epoch in range(num_epochs):
    # train process
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True)
    sum_loss = 0
    for i, data in enumerate(data_loader_train):
        inputs, ground_truth = data
        optimizer.zero_grad()
        inputs = torch.transpose(inputs, 3, 1)
        # print(inputs.shape)
        outputs = model(inputs.cuda().float())
        # print(outputs.shape)
        # print(ground_truth.shape)

        ground_truth = torch.transpose(ground_truth.cuda(), 2, 1)
        loss = criterion(outputs.reshape(outputs.shape[1:]), ground_truth.float())
        sum_loss += loss
        loss.backward()
        optimizer.step()
    average = sum_loss / i
    print(average)
    tag = "average loss each epoch".format(epoch + 1)
    writer.add_scalar(tag, average, epoch)

pkl_file_name = "final_epoch_{}_iou_{}.pkl".format(epoch,iou)
pkl_file_path = os.path.join(model_pth, pkl_file_name)
torch.save(model.state_dict(), pkl_file_path)
writer.close() 

# record end time
end_time = datetime.now()
print("end train: ", end_time)
print("time cost: {}s".format((end_time-begin_time).seconds))
