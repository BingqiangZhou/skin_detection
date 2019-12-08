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

data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True)

# indices = torch.randperm(len(dataset)).tolist()
# dataset_train = torch.utils.data.Subset(dataset, indices[:5])
# dataset_test = torch.utils.data.Subset(dataset, indices[5:])

# load our model and set directory for saving model parameter
model_pth = r'.\model_pth'

model = SkinDetectionNet()

is_train_in_gpu = True
if is_train_in_gpu == True:
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
model.eval()

# set loss function and optimizer
criterion = torch.nn.BCEWithLogitsLoss()
# criterion = torch.nn.BCELoss()
# criterion = torch.nn.L1Loss()
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.001)

# set how many epochs to train
num_epochs = 20

# record start time
begin_time = datetime.now()
print("begin train: ", begin_time)

# Starting train and validate.
for epoch in range(num_epochs):
    # train process
    sum_loss = 0
    for i, data in enumerate(data_loader_train):
        optimizer.zero_grad()

        input, ground_truth = data

        if is_train_in_gpu == True:
            input = input.cuda()
            ground_truth = ground_truth.cuda()

        # print(inputs.shape)
        output = model(input.float())
        # print(output.shape)
        # print(ground_truth.shape)
        output = output.squeeze(axis=0)
        loss = criterion(output, ground_truth.float())
        sum_loss += loss
        loss.backward()
        optimizer.step()

    average = sum_loss / i
    print(epoch + 1, average)
    tag = "average loss each epoch".format(epoch + 1)
    writer.add_scalar(tag, average, epoch)

pkl_file_name = "final_epoch_{}_{}.pkl".format(epoch+1, str(datetime.now()).replace(':','_'))
pkl_file_path = os.path.join(model_pth, pkl_file_name)
torch.save(model.state_dict(), pkl_file_path)
writer.close()

# record end time
end_time = datetime.now()
print("end train: ", end_time)
print("time cost: {}s".format((end_time-begin_time).seconds))
