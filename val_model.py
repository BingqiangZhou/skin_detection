import torch
import os
from my_model import SkinDetectionModel
from PIL import Image
import numpy as np
import  matplotlib.pyplot as plt
from datetime import datetime
import time

test_dir = r'.\dataset\test'
# pth_dir = r'.\model_pth\2019-11-24 21-15-06.966213\pth_files'
pth_dir = r'.\model_pth\2019-11-24 23-23-17.993480\pth_files'

model_pth_file = None

file_names = os.listdir(pth_dir)
for file_name in file_names:
    if file_name.find('final') != -1:
        model_pth_file = file_name
        break

assert model_pth_file != None, "can't find model file, please set the file's name include 'final'."

model_file_path = os.path.join(pth_dir,model_pth_file)

result_name_path = model_pth_file+"_"+str(datetime.now()).replace(':','-')
new_dir = os.path.join(test_dir,result_name_path)
os.mkdir(new_dir)


model = SkinDetectionModel()
model.eval()

print("loading model parameters……")
model.load_state_dict(torch.load(model_file_path))
print("loading model parameters has been finished.")
print("detecting image……")

images_path = os.path.join(test_dir,'images')
image_names = os.listdir(images_path)
for image_name in image_names:
    test_image_file_path = os.path.join(images_path,image_name)
    input = np.array(Image.open(test_image_file_path))

    start_time = time.clock()

    input = input[np.newaxis,:]
    input = torch.from_numpy(input)
    input = torch.transpose(input, 3, 1)
    output = model(input.float())
    output = output.view(output.shape[2:])
    output = torch.transpose(output, 0, 1)
    output[output > 0] = 1
    output[output < 0] = 0
    output = output.detach().numpy()

    end_time = time.clock()

    plt.imsave(os.path.join(new_dir,image_name), output)
    # plt.imshow(output)
    print("{}, time cost: {}s".format(test_image_file_path, end_time - start_time))
print("detecting image was done.")