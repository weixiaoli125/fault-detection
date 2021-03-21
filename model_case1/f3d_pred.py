# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:12:19 2020

@author: lyric
"""

import tensorflow as tf
import numpy as np
import os
# import cv2
from tqdm import tqdm
import math
import datetime,time
import matplotlib.pyplot as plt

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

predict_batch_size = 1024
dropout_rate = 0.5

log_dir = "./log"
model_path ="./log/521model.ckpt.meta"
index_path = "./log"
saver = tf.train.import_meta_graph(model_path)# 加载图结构
gragh = tf.get_default_graph()# 获取当前图，为了后续训练时恢复变量
tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]# 得到当前图中所有变量的名称

x = gragh.get_tensor_by_name('Placeholder:0')# 获取输入变量（占位符，由于保存时未定义名称，tf自动赋名称“Placeholder”）
y = gragh.get_tensor_by_name('Placeholder_1:0')# 获取输出变量
keep_prob = gragh.get_tensor_by_name('Placeholder_2:0')# 获取dropout的保留参数

softmax = gragh.get_tensor_by_name('softmax/softmax:0')
pred = gragh.get_tensor_by_name('softmax/Cast:0')# 获取网络输出值
fc_weight1 = gragh.get_tensor_by_name('fc1/fc_weight1:0')# 获取fc_weight1

test_x=np.load("bibubic_crossline_800_cut1.npy")
print('>>>test_x shape:',test_x.shape)

starttime = datetime.datetime.now()
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint(index_path))# 加载变量值
    print('finish loading model!') 
    #计算测试集精度
    test_predict = np.array([])
    test_logit = np.array([])
    logits = np.transpose(np.array([[],[]]))
    test_loss = 0
    for j in range(0, int(math.ceil(test_x.shape[0]/predict_batch_size))):
        print(j)
        predict_batch,logit_batch = sess.run([pred,softmax],feed_dict={x:test_x[j * predict_batch_size:(j + 1) * predict_batch_size],
                                                                            keep_prob:dropout_rate})

        print('<<<len(predict_batch):',len(predict_batch))
        print('<<<len(logit_batch):',logit_batch.shape)
        print('<<<len(logit_batch):',len(logit_batch[:,1]))
        test_predict = np.concatenate((test_predict, predict_batch))
        test_logit = np.concatenate((test_logit,logit_batch[:,1]))
        logits = np.concatenate((logits,logit_batch),axis=0)
    print('sussfully save test predict!')
    
endtime = datetime.datetime.now()
print('>>>test time:',(endtime - starttime).seconds)