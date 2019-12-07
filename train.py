import os
import torchvision
import torch
import  numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from dataset import SkinDetectionDataset
from my_model import SkinDetectionModel

# get our dataset
root_dir = r'.\dataset'
dataset = SkinDetectionDataset(root_dir)

indices = torch.randperm(len(dataset)).tolist()
dataset_train = torch.utils.data.Subset(dataset, indices[:5])
dataset_test = torch.utils.data.Subset(dataset, indices[5:])

data_loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=1, shuffle=True)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=True)

# load our model and set directory for saving model parameter
model_pth = r'.\model_pth'

model = SkinDetectionModel()
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
val_image_dir = os.path.join(pth_dir,"val_images") 
os.mkdir(val_image_dir)
pth_file_dir = os.path.join(pth_dir,"pth_files") 
os.mkdir(pth_file_dir)

# Starting train and validate.
for epoch in range(num_epochs):
    # train process
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
            loss.backward()
            optimizer.step()
    # print('Finished Training')

    # validate process
    for i, data in enumerate(data_loader_test):
        inputs, ground_truth = data
        inputs = torch.transpose(inputs.cuda(), 3, 1)
        outputs = model(inputs.float())
        outputs = outputs.view(outputs.shape[1:])
        outputs = torch.transpose(outputs, 2, 1)

        # calculate iou
        outputs[outputs > 0] = 1
        outputs[outputs < 0] = 0
        temp = outputs + ground_truth.cuda()
        intersection = torch.zeros_like(outputs)
        union = torch.zeros_like(outputs)
        # print(outputs.shape)
        intersection[temp >= 1] = 1
        union[temp == 2] = 1
        iou = torch.sum(union)/ torch.sum(intersection)
        print("epoch:{}, iou:{}".format(epoch,iou))

        # save model's parameter and output image at each 'num_epochs_save' step
        if (epoch + 1) % num_epochs_save == 0:
            pkl_file_name = "epoch_{}_iou_{}.pkl".format(epoch+1,iou)
            pkl_file_path = os.path.join(pth_file_dir,pkl_file_name)
            torch.save(model.state_dict(), pkl_file_path)
            # torch.save(model.state_dict(), ".\\model_pth\\epoch_{}_iou_{}.pkl".format(epoch,iou))

            # save output image
            output_image_name = "{}_{}.png".format(epoch+1, i)
            result_file_path = os.path.join(val_image_dir,output_image_name)
            outputs = outputs.cpu().detach().numpy()
            plt.imsave(result_file_path,outputs.reshape(outputs.shape[1:]))


        if iou >= accuracy_exceed_precent :
            accuracy_exceed_precent_flag = True
    
    # stop train and save model's parameter
    if (accuracy_exceed_precent_flag == True and is_allow_end_train == True) or epoch+1 == num_epochs :
        pkl_file_name = "final_epoch_{}_iou_{}.pkl".format(epoch,iou)
        pkl_file_path = os.path.join(pth_file_dir,pkl_file_name)
        torch.save(model.state_dict(), pkl_file_path)

        output_image_name = "final_epoch_{}_iou_{}.png".format(epoch, i)
        result_file_path = os.path.join(val_image_dir,output_image_name)
        if epoch+1 == 100:
            outputs = outputs.detach().numpy()
        plt.imsave(result_file_path,outputs.reshape(outputs.shape[1:]))
        break

# record end time
end_time = datetime.now()
print("end train: ", end_time)
print("time cost: {}s".format((end_time-begin_time).seconds))
