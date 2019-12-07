import os

from utils import combine_image_and_mask_from_file_path

image_dir = r'.\dataset\test\images'
mask_dir = r'.\dataset\test\final_epoch_99_iou_0.8547319769859314.pkl_2019-11-24 23-37-31.015469'

combine_image_and_mask_save_dir_name = r'result of combine iamge and mask'
combine_image_and_mask_save_dir = os.path.join(mask_dir,combine_image_and_mask_save_dir_name)
if os.path.exists(combine_image_and_mask_save_dir) == False:
    os.mkdir(combine_image_and_mask_save_dir)

image_names = os.listdir(image_dir)
for image_name in image_names:
    image_path = os.path.join(image_dir, image_name)
    mask_path = os.path.join(mask_dir, image_name)
    combine_image_and_mask_file_name = image_name.replace(".jpg",'.png')
    combine_image_and_mask_save_path = os.path.join(combine_image_and_mask_save_dir, combine_image_and_mask_file_name)
    combine_image_and_mask_from_file_path(image_path, mask_path, save_path=combine_image_and_mask_save_path,alpha=0.5)

