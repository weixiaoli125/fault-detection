# -*- coding: utf-8 -*-
"""
Created on Sat May 23 22:20:33 2020

@author: lyric
"""

import os
import numpy as np

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

#边界处理
# def clip_boundary(data):
#     return data[2:,2:]

# def get_patchs(image,label,size,step):
#     m,n=image.shape
#     image_patchs=[]
#     label_patchs=[]
#     for i in range(size//2,m-size//2,step):
#         for j in range(size//2,n-size//2,step):
#             if label[i,j]==0:
#                 image_patchs.append(image[(i-size//2):(i+size//2+1),(j-size//2):(j+size//2+1)][:,:,np.newaxis])
#                 label_patchs.append(0)

#     fault_id=np.argwhere(label==1)
#     for item in fault_id:
#         if (item[0]>=size//2 and item[0]<=m-size//2-1) and (item[1]>=size//2 and item[1]<=n-size//2-1):
#            image_patchs.append(image[(item[0]-size//2):(item[0]+size//2+1),(item[1]-size//2):(item[1]+size//2+1)][:,:,np.newaxis])
#            label_patchs.append(1)
#     return image_patchs,label_patchs

def get_patchs(image,label,size,step):
    m,n=image.shape
    image_patchs=[]
    label_patchs=[]
    for i in range(size//2,m-size//2,step):
        for j in range(size//2,n-size//2,step):
            if len(np.argwhere(label[(i-size//2):(i+size//2+1),(j-size//2):(j+size//2+1)]==1))==0:
            # if label[i][j] != 1:
                image_patchs.append(image[(i-size//2):(i+size//2+1),(j-size//2):(j+size//2+1)][:,:,np.newaxis])
                label_patchs.append(0)

    fault_id=np.argwhere(label==1)
    for item in fault_id:
        if (item[0]>=size//2 and item[0]<=m-size//2-1) and (item[1]>=size//2 and item[1]<=n-size//2-1):
          image_patchs.append(image[(item[0]-size//2):(item[0]+size//2+1),(item[1]-size//2):(item[1]+size//2+1)][:,:,np.newaxis])
          label_patchs.append(1)
    return image_patchs,label_patchs

def get_data_set(data_path,size):
#    all_name=os.listdir(os.path.join(data_path,'f'))#断层图像txt

    n = 800
    train_x=[]
    train_y=[]
    val_x=[]
    val_y=[]
    for i in range(n):
        print('processing image ',i+1)
        image_path=data_path +'/f_' +'/f_' +str(i+1)+'.txt'
        label_path=data_path +'/label_' +'/label_' +str(i+1)+'.txt'
        image= scale_process(cilp_image(load_image(image_path)))
        label= load_label(label_path)
        if i< int(n*0.8):
            image_patchs,label_patchs=get_patchs(image=image,label=label,size=size,step=2)#10,5,3,1,2
            train_x += image_patchs
            train_y += label_patchs
        elif i< int(n*0.9):
            image_patchs, label_patchs = get_patchs(image=image, label=label, size=size, step=29)#10,5,3,2,1
            val_x += image_patchs
            val_y += label_patchs
        # else:
        #     np.savetxt('./test_data/f_'+str(i+1),(image))
        #     np.savetxt('./test_data/label_'+str(i+1),(label))
            # image_patchs, label_patchs = get_patchs(image=image, label=label, size=size, step=3)
            # test_x += image_patchs
            # test_y += label_patchs


    train_x=np.array(train_x)
    np.random.seed(123)
    np.random.shuffle(train_x)
    train_y=np.array(train_y)
    np.random.seed(123)
    np.random.shuffle(train_y)

    val_x=np.array(val_x)
    np.random.seed(456)
    np.random.shuffle(val_x)
    val_y=np.array(val_y)
    np.random.seed(456)
    np.random.shuffle(val_y)

#     test_x=np.array(test_x)
# #    np.random.seed(789)
# #    np.random.shuffle(test_x)
#     test_y=np.array(test_y)
#    np.random.seed(789)
#    np.random.shuffle(test_y)

    return train_x,train_y,val_x,val_y


if __name__=="__main__":
    data_path="F:/wei_loss/datame/data_patches_huhe"
    size=45
    
    train_x, train_y, val_x, val_y =get_data_set(data_path=data_path,size=size)
    
    print('>>>train_x shape:',train_x.shape)
    print('>>>train_y shape:', train_y.shape)
    print('>>>positive instances:',np.sum(train_y==1),' negative instances:',np.sum(train_y==0))
    print()
    
    print('>>>val_x shape:',val_x.shape)
    print('>>>val_y shape:', val_y.shape)
    print('>>>positive instances:', np.sum(val_y == 1), ' negative instances:', np.sum(val_y == 0))
    print()

    # print('>>>test_x shape:',test_x.shape)
    # print('>>>test_y shape:', test_y.shape)
    # print('>>>positive instances:', np.sum(test_y == 1), ' negative instances:', np.sum(test_y == 0))
    # print()    
        
    print('start saving data...')

    np.save('F:/wei_loss/datame/data_patches_huhe/patches/train_x_step_2_.npy',train_x)
    np.save('F:/wei_loss/datame/data_patches_huhe/patches/train_y_step_2_.npy',train_y)

    # np.save('F:/wei_loss/datame/data_patches_huhe/patches/val_x_step_1_.npy',val_x)
    # np.save('F:/wei_loss/datame/data_patches_huhe/patches/val_y_step_1_.npy',val_y)

    # np.save('./data_patches/test_x.npy',test_x)
    # np.save('./data_patches/test_y.npy',test_y)

    print('saving data finished!')