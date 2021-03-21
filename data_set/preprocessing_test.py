# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 20:54:20 2020

@author: win10
"""


# -*- coding: utf-8 -*-
"""
Created on Sat May 23 22:20:33 2020

@author: lyric
"""

import os
import numpy as np
import cv2
#加载图像txt                                                   
def load_image(image_path):
    image=np.loadtxt(image_path,dtype=np.float32)
    return image

#将图像clip到-1到1之间
def cilp_image(image):
    return np.clip(image,-1,1)

#加载标签label
def load_label(label_path):
    label=np.loadtxt(label_path,dtype=np.int32)
    return label 

#Min-Max Normalization
def scale_process(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def padding_images(image):
    paddingimage = cv2.copyMakeBorder(image,22, 22, 22, 22 ,cv2.BORDER_REPLICATE)#cv2.BORDER_REPLICATE
    return paddingimage

def padding_labels(label):
    paddinglabel = cv2.copyMakeBorder(label,22, 22, 22, 22 ,0)#cv2.BORDER_CONSTANT
    return paddinglabel

def get_patchs(image,size,step):
    m,n=image.shape
    image_patchs=[]
    for i in range(size//2,m-size//2,step):
        for j in range(size//2,n-size//2,step):
            image_patchs.append(image[(i-size//2):(i+size//2+1),(j-size//2):(j+size//2+1)][:,:,np.newaxis])
    return image_patchs

def get_data_set(data_path,size):
#    all_name=os.listdir(os.path.join(data_path,'f'))#断层图像txt

    n = 800
    test_x=[]

    for i in range(720,n):
        print('processing test image ',i+1)
        image_path=data_path +'f_' +str(i+721)+'.txt'
        # label_path=data_path +'/label_' +str(i+641)
        image= padding_images(scale_process(cilp_image(load_image(image_path))))
        image_patchs= get_patchs(image=image, size=size, step=1)
        test_x += image_patchs


    test_x=np.array(test_x)

    return test_x


if __name__=="__main__":
    data_path="./"
    size=45
    
    test_x =get_data_set(data_path=data_path,size=size)
           
    print('start saving data...')

    np.save("test_x_step_1",test_x)

    print('saving data finished!')
    
    # import matplotlib.pyplot as plt
    # x = np.loadtxt(data_path+'/f_' +str(671)+'.txt')
    # plt.axis('off')
    # a=padding_images(scale_process(cilp_image(load_image(image_path))))
    # plt.imshow(x,cmap='gray')
