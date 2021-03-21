# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:10:02 2020

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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

predict_batch_size = 1024#67006
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
conv1 = gragh.get_tensor_by_name('conv1:0')
conv4 = gragh.get_tensor_by_name('conv4:0')
maxpool1 = gragh.get_tensor_by_name('conv3/pool1:0')
maxpool2 = gragh.get_tensor_by_name('conv6/pool2:0')
# 定义评价指标
# loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(y,tf.int32),logits=logit)#+tf.contrib.layers.l2_regularizer(0.01)(fc_weight1)
# test_accuracy=np.sum(pred ==y)/(np.sum(y==1)+np.sum(y==0)+1e-9)
# test_precision=np.sum((y==1)*(pred==1))/(np.sum(pred==1)+1e-9)
# test_recall=np.sum((y==1)*(pred==1))/(np.sum(y==1)+1e-9)
# test_f1_score=2*test_precision*test_recall/(test_precision+test_recall+1e-9)
test_x=np.load('./patches_set/test_x_step_1.npy')

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
                                          # y:test_y[j * predict_batch_size:(j + 1) * predict_batch_size],
                                                                            keep_prob:dropout_rate})

        print('<<<len(predict_batch):',len(predict_batch))
        print('<<<len(logit_batch):',logit_batch.shape)
        print('<<<len(logit_batch):',len(logit_batch[:,1]))
        test_predict = np.concatenate((test_predict, predict_batch))
        test_logit = np.concatenate((test_logit,logit_batch[:,1]))
        logits = np.concatenate((logits,logit_batch),axis=0)
        
    #     test_loss += loss_bath*len(test_y[j * predict_batch_size:(j + 1) * predict_batch_size])
    # test_loss = test_loss/len(test_predict)
    # print('<<<len(test):',len(test_predict))
    # test_accuracy=np.sum(test_predict==test_y)/(len(test_y)+1e-9)
    # test_precision=np.sum((test_y==1)*(test_predict==1))/(np.sum(test_predict==1)+1e-9)
    # test_recall=np.sum((test_y==1)*(test_predict==1))/(np.sum(test_y==1)+1e-9)
    # test_f1_score=2*test_precision*test_recall/(test_precision+test_recall+1e-9)
    # print('>>>test results:>>> test_loss:',test_loss,'test_accuracy:{:.2%}'.format(test_accuracy),' test_precision:',test_precision,' test_recall:',test_recall,' test_f1_score:',test_f1_score)
    # np.savetxt('./test_pred/logits.txt',logits)
    # np.savetxt('./test_pred/predict_test.txt',test_predict)
    # np.savetxt('./test_pred/logit_test.txt',test_logit)
    print('sussfully save test predict!')

            
        
endtime = datetime.datetime.now()
print('>>>test time:',(endtime - starttime).seconds)