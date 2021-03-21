# -*- coding: utf-8 -*-
"""
Created on Sat May 23 22:31:26 2020

@author: lyric
"""

import sys
import math
import numpy as np
np.random.seed(123)
import tensorflow as tf
from tensorflow.python.ops import array_ops
from Logger import Logger
from focal_loss import focal_loss,focal_loss_fixed,balanced_loss
from model import FaultDetection
#from lgm import *
import matplotlib.pyplot as plt
import matplotlib
tf.set_random_seed(123)
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
import os
import datetime,time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__=="__main__":
    sys.stdout = Logger("result_loss.txt" ) #将模型训练过程和结果保存到文件
    

    train_x=np.load("train_x_step_2.npy")
    train_y=np.load("train_y_step_2.npy")
    print('>>>train_x shape:',train_x.shape)
    print('>>>train_y shape:', train_y.shape)
    print('>>>positive instances:',np.sum(train_y==1),' negative instances:',np.sum(train_y==0))
    print()
    
    val_x=np.load("val_x_step_1.npy")
    val_y=np.load("val_y_step_1.npy")
    print('>>>val_x shape:',val_x.shape)
    print('>>>val_y shape:', val_y.shape)
    print('>>>positive instances:', np.sum(val_y == 1), ' negative instances:', np.sum(val_y == 0))
    print()
  
    model=FaultDetection()
    model.construct_network()
    model.train(train_x=train_x,train_y= train_y.astype(np.float32),val_x=val_x,val_y=val_y.astype(np.float32))
    ##显示训练Loss
    plt.figure(13)
    plt.plot(losses['train'], label='Training loss')
    plt.legend()
    _ = plt.ylim()
    plt.savefig('figure_train_loss.png')
    ##显示验证Loss
    plt.plot(losses['val'], label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    _ = plt.ylim()
    plt.savefig('figure_val_loss.png')


