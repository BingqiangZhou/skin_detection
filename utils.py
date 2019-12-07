# reference urls
# combine image and mask: https://blog.csdn.net/SenPaul/article/details/95360290
# numpy to PIL image: https://blog.csdn.net/liuweizj12/article/details/80221537
# cannot write mode RGBA as JPEG: https://github.com/python-pillow/Pillow/issues/2609

from PIL import Image
import matplotlib.pyplot as plt
 

def combine_image_and_mask_from_file_path(image_path,mask_path,*,save_path=None,alpha=0.3):
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    
    image = image.convert('RGBA')
    mask = mask.convert('RGBA')
    
    image = Image.blend(image,mask,alpha)
    if save_path != None :
        image.save(save_path)


def combine_image_and_mask_from_PIL_image(image,mask,alpha=0.3):
    image = image.convert('RGBA')
    mask = mask.convert('RGBA')
    
    image = Image.blend(image,mask,alpha)
    return image

def combine_image_and_mask_from_numpy_array(image,mask,alpha=0.3):

    image = Image.fromarray(image)
    mask = Image.fromarray(mask)

    image = image.convert('RGBA')
    mask = mask.convert('RGBA')
    
    image = Image.blend(image,mask,alpha)
    return image

# plt.imshow(image1)
# plt.imshow(image2,alpha=0.5)
 
# plt.show()
