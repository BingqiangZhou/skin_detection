import os
import numpy as np
import matplotlib.pyplot as plt

temp_images_dir = './dataset/temp'
ground_truths_dir = './dataset/groundtruths'

image_name_list = os.listdir(temp_images_dir)

for image_name in image_name_list:
    img = plt.imread(os.path.join(temp_images_dir,image_name))
    img = np.array(img)

    ground_truth_img = np.zeros(img.shape[:2])

    #print(img[img.shape[0]//2,img.shape[1]//2])
    #print(img.shape)
    #print(ground_truth.shape)

    #(image==[255,0,0]).all(axis=2)
    index = (np.array(img[:,:,0]>155,np.int)+np.array(img[:,:,2]<50,np.int)+np.array(img[:,:,1]<50,np.int) == 3)
    ground_truth_img[index] = 1
    
    ground_truth_file_name = image_name.replace(".jpg",".png")
    ground_truth_path = os.path.join(ground_truths_dir,ground_truth_file_name)
    plt.imsave(ground_truth_path,ground_truth_img)


