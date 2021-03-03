# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 17:40:06 2021

@author: win10
"""


import sys


import math
import numpy as np
np.random.seed(123)
import tensorflow as tf
from tensorflow.python.ops import array_ops
from Logger import Logger
from focal_loss import focal_loss,focal_loss_fixed,balanced_loss
#from lgm import *
import matplotlib.pyplot as plt
import matplotlib
tf.set_random_seed(123)
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
import os
import datetime,time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
losses = {'train':[], 'val':[]}  
model_path = "./results/xin_59/log/521model.ckpt"  
class FaultDetection:

    def __init__(self):

        self.graph = tf.get_default_graph()
        self.session = tf.Session(graph=self.graph)

        self.width=45
        self.height=45
        self.channel=1
        self.filter_num_1= 20#20
        self.filter_num_2= 50#40
        self.max_pool_size= 2
        self.filter_size=3
        self.fc_num_1= 32#10
        self.fc_num_2= 16#20
        self.class_num= 2

        self.epochs=50
        self.batch_size=30#1024#1024#30#
        self.learning_rate=0.00001#0.0001
        self.momentum=0.9
        self.weight_decay=0.01#0.05
        self.predict_batch_size=1024#4096#
        
        # self.values = [0.0001,0.00001]
        # self.boundaries = [20]

        # self.global_step = tf.Variable(0,trainable=False)  # 因为此代码没有管理global的优化器，其实此处没有用
        
        
        self.beta=0.5       

    def construct_network(self):
        self.input=tf.placeholder(dtype=tf.float32,shape=(None,self.width,self.height,self.channel))
        self.label=tf.placeholder(dtype=tf.float32,shape=(None))
        self.keep_prob = tf.placeholder("float")
        # self.learning_rate = tf.placeholder("float")
            
        with tf.name_scope("conv1") as scope:
            kernel1 = tf.Variable(tf.truncated_normal([self.filter_size,self.filter_size, self.channel,self.filter_num_1], mean=0, stddev=0.1,
                                                      dtype=tf.float32), trainable=True,name="weights")
            conv = tf.nn.conv2d(self.input, kernel1, [1,1,1,1], padding="SAME")
            biases = tf.Variable(tf.constant(0, shape=[self.filter_num_1], dtype=tf.float32), trainable=True, name="biases")
            bias1 = tf.nn.bias_add(conv, biases)
            bn1 = tf.contrib.layers.batch_norm(inputs=bias1,decay=0.9,updates_collections=None,is_training = True)
            self.conv1 = tf.nn.relu(bn1, name=scope)

        # with tf.name_scope("conv2") as scope:
        #     kernel2 = tf.Variable(tf.truncated_normal([self.filter_size,self.filter_size, self.filter_num_1,self.filter_num_1], mean=0, stddev=0.1,
        #                                               dtype=tf.float32), trainable=True,name="weights")
        #     conv = tf.nn.conv2d(bias1, kernel2, [1,1,1,1], padding="SAME")
        #     biases = tf.Variable(tf.constant(0, shape=[self.filter_num_1], dtype=tf.float32), trainable=True, name="biases")
        #     bias2 = tf.nn.bias_add(conv, biases)
        #     # bn2 = tf.contrib.layers.batch_norm(inputs=bias2,decay=0.9,updates_collections=None,is_training = True)
        #     self.conv2 = tf.nn.relu(bias2, name=scope)

        with tf.name_scope("conv3") as scope:
            kernel3 = tf.Variable(tf.truncated_normal([self.filter_size,self.filter_size, self.filter_num_1,self.filter_num_1], mean=0, stddev=0.1,
                                                      dtype=tf.float32), trainable=True,name="weights")
            conv = tf.nn.conv2d(self.conv1, kernel3, [1,1,1,1], padding="SAME")
            biases = tf.Variable(tf.constant(0, shape=[self.filter_num_1], dtype=tf.float32), trainable=True, name="biases")
            bias3 = tf.nn.bias_add(conv, biases)
            bn3 = tf.contrib.layers.batch_norm(inputs=bias3,decay=0.9,updates_collections=None,is_training = True)
            self.conv3 = tf.nn.relu(bn3, name=scope)
            # activation1 = tf.nn.relu(self.conv3, name='activation1')
            self.max_pool1=tf.nn.max_pool(self.conv3,ksize=[1,self.max_pool_size,self.max_pool_size,1],strides=[1,2,2,1],padding="VALID",name="pool1")
            # lrn3 = tf.nn.lrn(max_pool1,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='nornl')
            activation2 = tf.nn.relu(self.max_pool1, name='activation2')
            
        with tf.name_scope("conv4") as scope:
            kernel4 = tf.Variable(tf.truncated_normal([self.filter_size,self.filter_size, self.filter_num_1,self.filter_num_2], mean=0, stddev=0.1,
                                                      dtype=tf.float32), trainable=True,name="weights")
            conv = tf.nn.conv2d(activation2, kernel4, [1,1,1,1], padding="SAME")
            biases = tf.Variable(tf.constant(0, shape=[self.filter_num_2], dtype=tf.float32), trainable=True, name="biases")
            bias4 = tf.nn.bias_add(conv, biases)
            bn4 = tf.contrib.layers.batch_norm(inputs=bias4,decay=0.9,updates_collections=None,is_training = True)
            self.conv4 = tf.nn.relu(bn4, name=scope)


        # with tf.name_scope("conv5") as scope:
        #     kernel5 = tf.Variable(tf.truncated_normal([self.filter_size,self.filter_size, self.filter_num_2,self.filter_num_2], mean=0, stddev=0.1,
        #                                               dtype=tf.float32), trainable=True,name="weights")
        #     conv = tf.nn.conv2d(self.conv4, kernel5, [1,1,1,1], padding="SAME")
        #     biases = tf.Variable(tf.constant(0, shape=[self.filter_num_2], dtype=tf.float32), trainable=True, name="biases")
        #     bias5 = tf.nn.bias_add(conv, biases)
        # #     bn5 = tf.contrib.layers.batch_norm(inputs=bias5,decay=0.9,updates_collections=None,is_training = True)
        #     conv5 = tf.nn.relu(bias5, name=scope)

        with tf.name_scope("conv6") as scope:
            kernel6 = tf.Variable(tf.truncated_normal([self.filter_size,self.filter_size, self.filter_num_2,self.filter_num_2], mean=0, stddev=0.1,
                                                      dtype=tf.float32), trainable=True,name="weights")
            conv = tf.nn.conv2d(self.conv4, kernel6, [1,1,1,1], padding="SAME")
            biases = tf.Variable(tf.constant(0, shape=[self.filter_num_2], dtype=tf.float32), trainable=True, name="biases")
            bias6 = tf.nn.bias_add(conv, biases)
            bn6 = tf.contrib.layers.batch_norm(inputs=bias6,decay=0.9,updates_collections=None,is_training = True)
            self.conv6 = tf.nn.relu(bn6, name=scope)
            # activation3 = tf.nn.relu(bn6, name='activation3')
            self.max_pool2=tf.nn.max_pool(self.conv6,ksize=[1,self.max_pool_size,self.max_pool_size,1],strides=[1,2,2,1],padding="VALID",name="pool2")
            activation4 = tf.nn.relu(self.max_pool2, name='activation4')

        with tf.name_scope("fc1") as scope:
            flatten = tf.reshape(activation4, (-1, activation4.shape[-1]*activation4.shape[-2]*activation4.shape[-3]))
            self.fc_weight1 = tf.Variable(tf.truncated_normal([tf.cast(flatten.shape[-1],tf.int32), self.fc_num_1], mean=0, stddev=0.1, dtype=tf.float32),trainable=True,name="fc_weight1")
            fc_bias1 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[self.fc_num_1]), trainable=True, name="fc_bias1")
            fc1 = tf.add(tf.matmul(flatten, self.fc_weight1), fc_bias1,name=scope)
            activation5 = tf.nn.relu(fc1, name='activation5')
            drop_out=tf.nn.dropout(activation5,self.keep_prob)

        with tf.name_scope("fc2") as scope:
            fc_weight2 = tf.Variable(tf.truncated_normal([self.fc_num_1, self.fc_num_2], mean=0, stddev=0.1, dtype=tf.float32),trainable=True,name="fc_weight2")
            fc_bias2 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[self.fc_num_2]), trainable=True, name="fc_bias2")
            fc2 = tf.add(tf.matmul(drop_out, fc_weight2), fc_bias2,name=scope)
            self.activation6 = tf.nn.relu(fc2, name='activation6')

        with tf.name_scope("softmax") as scope:
            # self.logit = tf.nn.softmax(tf.cast(activation6,tf.float32))
            softmax_weight = tf.Variable(tf.truncated_normal([self.fc_num_2, self.class_num], mean=0, stddev=0.1, dtype=tf.float32),
                                      trainable=True, name="softmax_weight")
            softmax_bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[self.class_num]), trainable=True, name="softmax_bias")
            self.logit = tf.add(tf.matmul(tf.cast(self.activation6,tf.float32), softmax_weight) , softmax_bias,name=scope)
            # print('self.logit:')
            # print(tf.shape(self.logit))
            # print('self.activation6：')
            # print(tf.shape(self.activation6))
            # # print(tf.shape(self.label))
            # print('self.predict:')s
            self.softmax =tf.nn.softmax(self.logit,name='softmax') #tf.nn.sigmoid(self.logit,name='sigmoid')#
            self.predict=tf.cast(tf.argmax(self.softmax, 1),tf.int32)
            # print('one_hot_self.la1bel:')
            # print(tf.shape(tf.one_hot(tf.cast(self.label,tf.int32),2,1,0,-1)))
            # print(':')
# tf.nn.softmax(self.logit)
    def train(self,train_x,train_y,val_x,val_y):
        # self.loss=tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(self.label,tf.int32),logits=self.logit)#+tf.contrib.layers.l2_regularizer(self.weight_decay)(self.fc_weight1)
        #self.loss=tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=tf.cast(self.logit,tf.float32),targets=tf.cast(tf.one_hot(tf.cast(self.label,tf.int32),2,1,0,-1),tf.float32),pos_weight=tf.cast(24,tf.float32)))
        #self.loss = focal_loss(y_pred=tf.nn.softmax(self.logit), y_true=tf.one_hot(tf.cast(self.label,tf.int32),2,1,0,-1),gamma=2,alpha=1)#+tf.contrib.layers.l2_regularizer(self.weight_decay)(self.fc_weight1)
        self.loss=focal_loss(y_pred=self.softmax, y_true=tf.one_hot(tf.cast(self.label,tf.int32),2,1,0,-1),gamma=3,alpha=0.5)#+tf.contrib.layers.l2_regularizer(self.weight_decay)(self.fc_weight1)
        # self.loss = focal_loss_fixed(y_pred=self.softmax, y_true=tf.one_hot(tf.cast(self.label,tf.int32),2,1,0,-1),gamma=2,alpha=0.25)
        # self.loss = balanced_loss(y_pred=self.softmax, y_true=tf.one_hot(tf.cast(self.label,tf.int32),2,1,0,-1),alpha=0.96)#96
        
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.predict,tf.float32), self.label), tf.float32))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)        
        # self.optimizer=tf.train.MomentumOptimizer(self.learning_rate,self.momentum).minimize(self.loss)

        self.init_op = tf.global_variables_initializer()
        self.init_local_op = tf.local_variables_initializer()
        
        self.saver = tf.train.Saver()
        
        with self.session as sess:
            sess.run(self.init_op)
            sess.run(self.init_local_op)

            for epoch in range(1,self.epochs+1):
                # self.learning_rate = tf.train.piecewise_constant(epoch,self.boundaries,self.values)
                # for i in range(0, (train_x.shape[0] // self.batch_size)):
                #     _=sess.run(self.optimizer,feed_dict={self.input:train_x[i * self.batch_size:(i + 1) * self.batch_size],
                #                                                                         self.label:train_y[i * self.batch_size:(i + 1) * self.batch_size],
                #                                                                                   self.keep_prob:0.5})
                    
                ### 由于训练集和测试集的tensor太大，所以按批次计算精度
                # 计算训练集精度
                np.random.seed(epoch)
                np.random.shuffle(train_x)
                np.random.seed(epoch)
                np.random.shuffle(train_y)
                
                np.random.seed(123+epoch)
                np.random.shuffle(val_x)
                np.random.seed(123+epoch)
                np.random.shuffle(val_y)
            
                train_loss=0
                train_predict = np.array([])
                train_logit = np.array([])
                
                for i in range(0, int(math.ceil(train_x.shape[0] / self.batch_size))):
                    _,loss,predict,logit =sess.run([self.optimizer,self.loss,self.predict,self.softmax],feed_dict={self.input:train_x[i * self.batch_size:(i + 1) * self.batch_size],
                                                                                        self.label:train_y[i * self.batch_size:(i + 1) * self.batch_size],
                                                                                                 self.keep_prob:0.5})
                    
                    train_loss+=loss*len(train_y[i * self.batch_size:(i + 1) * self.batch_size])
                    train_predict = np.concatenate((train_predict, predict))
                    train_logit = np.concatenate((train_logit,logit[:,1]))
                # print(':')
                # print(tf.shape(train_loss))
                # print(':')
                train_loss=train_loss/len(train_predict)
                losses['train'].append(train_loss)
                train_accuracy=np.sum(train_predict==train_y)/len(train_y)
                train_precision=np.sum((train_y==1)*(train_predict==1))/(np.sum(train_predict==1)+1e-9)
                train_recall=np.sum((train_y==1)*(train_predict==1))/(np.sum(train_y==1)+1e-9)
                train_f1_score=2*train_precision*train_recall/(train_precision+train_recall+1e-9)
                train_specificity = np.sum((train_y==0)*(train_predict==0))/(np.sum(train_y==0)+1e-9)
                #计算测试集精度
                val_predict=np.array([])
                val_logit=np.array([])
                val_loss=0
                #test_predict=[]
                for j in range(0, int(math.ceil(val_x.shape[0]/self.predict_batch_size))):
                    _,loss_,predict_,logit_=sess.run([self.optimizer,self.loss,self.predict,self.softmax],feed_dict={self.input:val_x[j * self.predict_batch_size:(j + 1) * self.predict_batch_size],
                                                                self.label:val_y[j * self.predict_batch_size:(j + 1) * self.predict_batch_size],
                                                                                self.keep_prob:0.5})
                    val_loss+=loss_*len(val_y[j * self.predict_batch_size:(j + 1) * self.predict_batch_size])
                    val_predict = np.concatenate((val_predict, predict_))
                    val_logit = np.concatenate((val_logit,logit_[:,1]))
                val_loss=val_loss/len(val_predict)
                losses['val'].append(val_loss)
                val_accuracy=np.sum(val_predict==val_y)/(len(val_y)+1e-9)
                val_precision=np.sum((val_y==1)*(val_predict==1))/(np.sum(val_predict==1)+1e-9)
                val_recall=np.sum((val_y==1)*(val_predict==1))/(np.sum(val_y==1)+1e-9)
                val_f1_score=2*val_precision*val_recall/(val_precision+val_recall+1e-9)
                val_specificity = np.sum((val_y==0)*(val_predict==0))/(np.sum(val_y==0)+1e-9)

                print('>>>Epoch:',epoch,'/',self.epochs,'\r\n',' train_loss:',train_loss,' train_accuracy:{:.2%}'.format(train_accuracy),' train_precision:',train_precision,' train_recall:',train_recall,' train_f1_score:',train_f1_score,'train_specificity:',train_specificity,
                      '\r\n',' val_loss:',val_loss, ' val_accuracy:{:.2%}'.format(val_accuracy),' val_precision:',val_precision,' val_recall:',val_recall,' val_f1_score:',val_f1_score,'val_specificity:',val_specificity)
            np.savetxt('./results/xin_59/predict_val.txt',(val_predict))
            np.savetxt('./results/xin_59/logit_val.txt',(val_logit)) 
            np.savetxt('./results/xin_59/predict_train.txt',(train_predict))
            np.savetxt('./results/xin_59/logit_train.txt',(train_logit))  
            # Save model weights to disk
            save_path = self.saver.save(sess, model_path)
            print("Model saved in file: %s" % save_path)

    #----------------------------------各个层特征可视化-------------------------------

            # imput image of nonfault
            fig = plt.figure(figsize = (8,8))
            A = train_x[train_y==0]
            #将横纵坐标都映射到（0，1）的范围内
            extent=(0,1,0,1)
            
            vmin=np.min(A)
            vmax=np.max(A)
            #将颜色映射到 vmin~vmax 之间
            norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
            h3 = plt.contourf(A[13200,:,:,0],cmap = plt.cm.GnBu,norm = norm,aspect='auto')#A[13200,:,:,0]
            # plt.colorbar()
            plt.axis('off')

            l = 0.92
            b = 0.12
            w = 0.015
            h = 1 - 2*b 
            
            #对应 l,b,w,h；设置colorbar位置；
            rect = [l,b,w,h] 
            cbar_ax = fig.add_axes(rect) 
            cb = plt.colorbar(h3, cax=cbar_ax)
            
            #设置colorbar标签字体等
            cb.ax.tick_params(labelsize=16)  #设置色标刻度字体大小。
            font = {'family' : 'serif',
            #       'color'  : 'darkred',
                'color'  : 'black',
                'weight' : 'normal',
                'size'   : 16,
                }
            cb.set_label('T' ,fontdict=font)            
            plt.savefig("F:/wei_loss/datame/results/xin_59/valization/nonfault.png",bbox_inches="tight", pad_inches=0.0)
            # plt.title('Fault patch:45x45')
            plt.show()
            
            # 第一层的卷积输出的特征图
            input_image = A[13200,:,:,0]
            input_image = np.expand_dims(input_image, axis=0)
            input_image = np.expand_dims(input_image, axis=3)
            conv1_20 = sess.run(self.conv1, feed_dict={self.input:input_image})     # [1, 45, 45 ,20] 
            conv1_transpose = sess.run(tf.transpose(conv1_20, [3, 0, 1, 2]))
            fig = plt.figure(figsize = (10,8))
            #将横纵坐标都映射到（0，1）的范围内
            extent=(0,1,0,1)
            
            vmin = np.min(conv1_transpose)
            vmax = np.max(conv1_transpose)
            #将颜色映射到 vmin~vmax 之间
            norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)#conv1_transpose[i][0]
            for i in range(20):
                plt.subplot(4,5,i+1)
                h3 = plt.contourf(conv1_transpose[i][0],cmap = plt.cm.GnBu,norm = norm)
                plt.axis('off')
            fig.subplots_adjust(right=0.9)
            l = 0.92
            b = 0.12
            w = 0.015
            h = 1 - 2*b 
            
            #对应 l,b,w,h；设置colorbar位置；
            rect = [l,b,w,h] 
            cbar_ax = fig.add_axes(rect) 
            cb = plt.colorbar(h3, cax=cbar_ax)
            
            #设置colorbar标签字体等
            cb.ax.tick_params(labelsize=16)  #设置色标刻度字体大小。
            font = {'family' : 'serif',
            #       'color'  : 'darkred',
                'color'  : 'black',
                'weight' : 'normal',
                'size'   : 16,
                }
            cb.set_label('T' ,fontdict=font)  
            plt.savefig("F:/wei_loss/datame/results/xin_59/valization/conv1.png",bbox_inches="tight", pad_inches=0.0)
            plt.show()
            
            # 第一层池化后的特征图
            pool1_20 = sess.run(self.max_pool1, feed_dict={self.input:input_image})     # [1, 22, 22, 20]
            pool1_transpose = sess.run(tf.transpose(pool1_20, [3, 0, 1, 2]))
            fig = plt.figure(figsize = (10,8))
            #将横纵坐标都映射到（0，1）的范围内
            extent=(0,1,0,1)
            
            vmin=np.min(pool1_transpose)
            vmax=np.max(pool1_transpose)
            #将颜色映射到 vmin~vmax 之间
            norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)#conv1_transpose[i][0]
            for i in range(20):
                plt.subplot(4,5,i+1)
                h3 = plt.contourf(pool1_transpose[i][0],cmap = plt.cm.GnBu, norm = norm)#
                plt.axis('off')
            fig.subplots_adjust(right=0.9)
            l = 0.92
            b = 0.12
            w = 0.015
            h = 1 - 2*b 
            
            #对应 l,b,w,h；设置colorbar位置；
            rect = [l,b,w,h] 
            cbar_ax = fig.add_axes(rect) 
            cb = plt.colorbar(h3, cax=cbar_ax)
            
            #设置colorbar标签字体等
            cb.ax.tick_params(labelsize=16)  #设置色标刻度字体大小。
            font = {'family' : 'serif',
            #       'color'  : 'darkred',
                'color'  : 'black',
                'weight' : 'normal',
                'size'   : 16,
                }
            cb.set_label('T' ,fontdict=font)  
            plt.savefig("F:/wei_loss/datame/results/xin_59/valization/pool1.png",bbox_inches="tight", pad_inches=0.0) 
            # plt.title('MaxPool1:20x22x22')
            plt.show()
            
            # 第4层卷积输出特征图
            conv3_50 = sess.run(self.conv4, feed_dict={self.input:input_image})          # [1, 22, 22, 50]
            conv3_transpose = sess.run(tf.transpose(conv3_50, [3, 0, 1, 2]))
            fig = plt.figure(figsize = (5,10))
            #将横纵坐标都映射到（0，1）的范围内
            extent=(0,1,0,1)
            
            vmin=np.min(conv3_transpose)
            vmax=np.max(conv3_transpose)
            #将颜色映射到 vmin~vmax 之间
            norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)#conv1_transpose[i][0]
            for i in range(50):
                plt.subplot(10,5,i+1)
                h3 = plt.contourf(conv3_transpose[i][0],cmap = plt.cm.GnBu, norm = norm)
                plt.axis('off')
            fig.subplots_adjust(right=0.9)
            plt.axis('off')
            l = 0.92
            b = 0.12
            w = 0.015
            h = 1 - 2*b 
            
            #对应 l,b,w,h；设置colorbar位置；
            rect = [l,b,w,h] 
            cbar_ax = fig.add_axes(rect) 
            cb = plt.colorbar(h3, cax=cbar_ax)
            
            #设置colorbar标签字体等
            cb.ax.tick_params(labelsize=16)  #设置色标刻度字体大小。
            font = {'family' : 'serif',
            #       'color'  : 'darkred',
                'color'  : 'black',
                'weight' : 'normal',
                'size'   : 16,
                }
            cb.set_label('T' ,fontdict=font)  
            plt.savefig("F:/wei_loss/datame/results/xin_59/valization/conv3.png",bbox_inches="tight", pad_inches=0.0)
            plt.show()
            
            # 第二层池化后的特征图
            pool2_50 = sess.run(self.max_pool2, feed_dict={self.input:input_image})         #[1, 11, 11, 50]
            pool2_transpose = sess.run(tf.transpose(pool2_50, [3, 0, 1, 2]))
            fig = plt.figure(figsize = (5,10))
            #将横纵坐标都映射到（0，1）的范围内
            extent=(0,1,0,1)
            
            vmin=np.min(pool2_transpose)
            vmax=np.max(pool2_transpose)
            #将颜色映射到 vmin~vmax 之间
            norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)#conv1_transpose[i][0]
            for i in range(50):
                plt.subplot(10,5,i+1)
                h3 = plt.contourf(pool2_transpose[i][0],cmap = plt.cm.GnBu, norm = norm)
                plt.axis('off')
            fig.subplots_adjust(right=0.9)
            plt.axis('off')
            l = 0.92
            b = 0.12
            w = 0.015
            h = 1 - 2*b 
            
            #对应 l,b,w,h；设置colorbar位置；
            rect = [l,b,w,h] 
            cbar_ax = fig.add_axes(rect) 
            cb = plt.colorbar(h3, cax=cbar_ax)
            
            #设置colorbar标签字体等
            cb.ax.tick_params(labelsize=16)  #设置色标刻度字体大小。
            font = {'family' : 'serif',
            #       'color'  : 'darkred',
                'color'  : 'black',
                'weight' : 'normal',
                'size'   : 16,
                }
            cb.set_label('T' ,fontdict=font) 
            plt.savefig("F:/wei_loss/datame/results/xin_59/valization/pool2.png",bbox_inches="tight", pad_inches=0.0)
            plt.show()
            
            # imput image of fault--------------------------------------------------------------------------
            fig = plt.figure(figsize = (20,20))
            B = train_x[train_y==1]
            #将横纵坐标都映射到（0，1）的范围内
            extent=(0,1,0,1)
            
            vmin=np.min(B)
            vmax=np.max(B)
            #将颜色映射到 vmin~vmax 之间
            norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
            h3 = plt.contourf(B[200,:,:,0],cmap = plt.cm.GnBu, norm = norm)
            plt.axis('off')

            l = 0.92
            b = 0.12
            w = 0.015
            h = 1 - 2*b 
            
            #对应 l,b,w,h；设置colorbar位置；
            rect = [l,b,w,h] 
            cbar_ax = fig.add_axes(rect) 
            cb = plt.colorbar(h3, cax=cbar_ax)
            
            #设置colorbar标签字体等
            cb.ax.tick_params(labelsize=16)  #设置色标刻度字体大小。
            font = {'family' : 'serif',
            #       'color'  : 'darkred',
                'color'  : 'black',
                'weight' : 'normal',
                'size'   : 16,
                }
            cb.set_label('T' ,fontdict=font)  
            plt.savefig("F:/wei_loss/datame/results/xin_59/valization/fault.png",bbox_inches="tight", pad_inches=0.0)
            # plt.title('Nonfault patch:45x45')
            plt.show()
            
            # 第一层的卷积输出的特征图
            input_image2 = B[200,:,:,0]
            input_image2 = np.expand_dims(input_image2, axis=0)
            input_image2 = np.expand_dims(input_image2, axis=3)
            conv1_20_non = sess.run(self.conv1, feed_dict={self.input:input_image2})     # [1, 45, 45 ,20] 
            conv1_transpose_non = sess.run(tf.transpose(conv1_20_non, [3, 0, 1, 2]))
            fig = plt.figure(figsize = (10,8))
            #将横纵坐标都映射到（0，1）的范围内
            extent=(0,1,0,1)
            
            vmin=np.min(conv1_transpose_non)
            vmax=np.max(conv1_transpose_non)
            #将颜色映射到 vmin~vmax 之间
            norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)#
            for i in range(20):
                plt.subplot(4,5,i+1)
                h3 = plt.contourf(conv1_transpose_non[i][0],cmap = plt.cm.GnBu, norm = norm)
                plt.axis('off')
            fig.subplots_adjust(right=0.9)
            l = 0.92
            b = 0.12
            w = 0.015
            h = 1 - 2*b 
            
            #对应 l,b,w,h；设置colorbar位置；
            rect = [l,b,w,h] 
            cbar_ax = fig.add_axes(rect) 
            cb = plt.colorbar(h3, cax=cbar_ax)
            
            #设置colorbar标签字体等
            cb.ax.tick_params(labelsize=16)  #设置色标刻度字体大小。
            font = {'family' : 'serif',
            #       'color'  : 'darkred',
                'color'  : 'black',
                'weight' : 'normal',
                'size'   : 16,
                }
            cb.set_label('T' ,fontdict=font)
            plt.savefig("F:/wei_loss/datame/results/xin_59/valization/fault_conv1.png",bbox_inches="tight", pad_inches=0.0)
            # plt.title('Conv1:20x45x45')
            plt.show()
            
            # 第一层池化后的特征图
            pool1_20_non = sess.run(self.max_pool1, feed_dict={self.input:input_image2})     # [1, 22, 22, 20]
            pool1_transpose_non = sess.run(tf.transpose(pool1_20_non, [3, 0, 1, 2]))
            fig = plt.figure(figsize = (10,8))
            #将横纵坐标都映射到（0，1）的范围内
            extent=(0,1,0,1)
            
            vmin=np.min(pool1_transpose_non)
            vmax=np.max(pool1_transpose_non)
            #将颜色映射到 vmin~vmax 之间
            norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)#
            for i in range(20):
                plt.subplot(4,5,i+1)
                h3 = plt.contourf(pool1_transpose_non[i][0],cmap = plt.cm.GnBu, norm = norm)
                plt.axis('off')
            fig.subplots_adjust(right=0.9)
            l = 0.92
            b = 0.12
            w = 0.015
            h = 1 - 2*b 
            
            #对应 l,b,w,h；设置colorbar位置；
            rect = [l,b,w,h] 
            cbar_ax = fig.add_axes(rect) 
            cb = plt.colorbar(h3, cax=cbar_ax)
            
            #设置colorbar标签字体等
            cb.ax.tick_params(labelsize=16)  #设置色标刻度字体大小。
            font = {'family' : 'serif',
            #       'color'  : 'darkred',
                'color'  : 'black',
                'weight' : 'normal',
                'size'   : 16,
                }
            cb.set_label('T' ,fontdict=font)
            plt.savefig("F:/wei_loss/datame/results/xin_59/valization/fault_pool1.png",bbox_inches="tight", pad_inches=0.0)
            plt.show()
            
            # 第3层卷积输出特征图
            conv3_50_non = sess.run(self.conv4, feed_dict={self.input:input_image2})          # [1, 22, 22, 50]
            conv3_transpose_non = sess.run(tf.transpose(conv3_50_non, [3, 0, 1, 2]))
            fig = plt.figure(figsize = (5,10))
            #将横纵坐标都映射到（0，1）的范围内
            extent=(0,1,0,1)
            
            vmin=np.min(conv3_transpose_non)
            vmax=np.max(conv3_transpose_non)
            #将颜色映射到 vmin~vmax 之间
            norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)#
            for i in range(50):
                plt.subplot(10,5,i+1)
                h3 = plt.contourf(conv3_transpose_non[i][0],cmap = plt.cm.GnBu, norm = norm)
                plt.axis('off')
            fig.subplots_adjust(right=0.9)
            l = 0.92
            b = 0.12
            w = 0.015
            h = 1 - 2*b 
            
            #对应 l,b,w,h；设置colorbar位置；
            rect = [l,b,w,h] 
            cbar_ax = fig.add_axes(rect) 
            cb = plt.colorbar(h3, cax=cbar_ax)
            
            #设置colorbar标签字体等
            cb.ax.tick_params(labelsize=16)  #设置色标刻度字体大小。
            font = {'family' : 'serif',
            #       'color'  : 'darkred',
                'color'  : 'black',
                'weight' : 'normal',
                'size'   : 16,
                }
            cb.set_label('T' ,fontdict=font)
            plt.savefig("F:/wei_loss/datame/results/xin_59/valization/fault_conv3.png",bbox_inches="tight", pad_inches=0.0)
            plt.show()
            
            # 第二层池化后的特征图
            pool2_50_non = sess.run(self.max_pool2, feed_dict={self.input:input_image2})         #[1, 11, 11, 50]
            pool2_transpose_non = sess.run(tf.transpose(pool2_50_non, [3, 0, 1, 2]))
            fig = plt.figure(figsize = (5,10))
            #将横纵坐标都映射到（0，1）的范围内
            extent=(0,1,0,1)
            
            vmin=np.min(pool2_transpose_non)
            vmax=np.max(pool2_transpose_non)
            #将颜色映射到 vmin~vmax 之间
            norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)#
            for i in range(50):
                plt.subplot(10,5,i+1)
                h3 = plt.contourf(pool2_transpose_non[i][0],cmap = plt.cm.GnBu, norm = norm)
                plt.axis('off')
            fig.subplots_adjust(right=0.9)
            l = 0.92
            b = 0.12
            w = 0.015
            h = 1 - 2*b 
            
            #对应 l,b,w,h；设置colorbar位置；
            rect = [l,b,w,h] 
            cbar_ax = fig.add_axes(rect) 
            cb = plt.colorbar(h3, cax=cbar_ax)
            
            #设置colorbar标签字体等
            cb.ax.tick_params(labelsize=16)  #设置色标刻度字体大小。
            font = {'family' : 'serif',
            #       'color'  : 'darkred',
                'color'  : 'black',
                'weight' : 'normal',
                'size'   : 16,
                }
            cb.set_label('T' ,fontdict=font)
            plt.savefig("F:/wei_loss/datame/results/xin_59/valization/fault_pool2.png",bbox_inches="tight", pad_inches=0.0)
            plt.show()
