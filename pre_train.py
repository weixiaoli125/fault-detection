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
from focal_loss import focal_loss,balanced_loss
import matplotlib.pyplot as plt
import matplotlib
tf.set_random_seed(123)
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
import os
import datetime,time



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
losses = {'train':[], 'val':[]}  

path ="./case2/focalloss/"
model_path = path+"log/521model.ckpt"  
class FaultDetection:

    def __init__(self):

        self.graph = tf.get_default_graph()
        self.session = tf.Session(graph=self.graph)

        self.width=45
        self.height=45
        self.channel=1
        self.filter_num_1= 20
        self.filter_num_2= 50
        self.max_pool_size= 2
        self.filter_size=3
        self.fc_num_1= 32
        self.fc_num_2= 16
        self.class_num= 2

        self.epochs=50
        self.batch_size=1024
        self.learning_rate=0.0001
        self.predict_batch_size=1024
              

    def construct_network(self):
        self.input=tf.placeholder(dtype=tf.float32,shape=(None,self.width,self.height,self.channel))
        self.label=tf.placeholder(dtype=tf.float32,shape=(None))
        self.keep_prob = tf.placeholder("float")
        self.is_training =  tf.placeholder(tf.bool,name='is_training')

            
        with tf.name_scope("conv1") as scope:
            kernel1 = tf.Variable(tf.truncated_normal([self.filter_size,self.filter_size, self.channel,self.filter_num_1], mean=0, stddev=0.1,
                                                      dtype=tf.float32), trainable=True,name="weights")
            conv = tf.nn.conv2d(self.input, kernel1, [1,1,1,1], padding="SAME")
            biases = tf.Variable(tf.constant(0, shape=[self.filter_num_1], dtype=tf.float32), trainable=True, name="biases")
            bias1 = tf.nn.bias_add(conv, biases)
            # bn1 = tf.layers.batch_normalization(bias1,training=True,reuse=tf.AUTO_REUSE) 
            bn1 = tf.contrib.layers.batch_norm(inputs=bias1,decay=0.9,updates_collections=None,is_training = self.is_training)
            self.conv1 = tf.nn.relu(bn1, name=scope)

        
        with tf.name_scope("conv2") as scope:
            kernel2 = tf.Variable(tf.truncated_normal([self.filter_size,self.filter_size, self.filter_num_1,self.filter_num_1], mean=0, stddev=0.1,
                                                      dtype=tf.float32), trainable=True,name="weights")
            conv = tf.nn.conv2d(self.conv1, kernel2, [1,1,1,1], padding="SAME")
            biases = tf.Variable(tf.constant(0, shape=[self.filter_num_1], dtype=tf.float32), trainable=True, name="biases")
            bias2 = tf.nn.bias_add(conv, biases)
            # bn3 = tf.layers.batch_normalization(bias3,training=True,reuse=tf.AUTO_REUSE) 
            bn2= tf.contrib.layers.batch_norm(inputs=bias2,decay=0.9,updates_collections=None,is_training = self.is_training)
            self.conv2 = tf.nn.relu(bn2, name=scope)
            self.max_pool1=tf.nn.max_pool(self.conv2,ksize=[1,self.max_pool_size,self.max_pool_size,1],strides=[1,2,2,1],padding="VALID",name="pool1")
            activation1 = tf.nn.relu(self.max_pool1, name='activation1')
            
 
        with tf.name_scope("conv3") as scope:
            kernel3 = tf.Variable(tf.truncated_normal([self.filter_size,self.filter_size, self.filter_num_1,self.filter_num_2], mean=0, stddev=0.1,
                                                      dtype=tf.float32), trainable=True,name="weights")
            conv = tf.nn.conv2d(activation1, kernel3, [1,1,1,1], padding="SAME")
            biases = tf.Variable(tf.constant(0, shape=[self.filter_num_2], dtype=tf.float32), trainable=True, name="biases")
            bias3 = tf.nn.bias_add(conv, biases)
            # bn4 = tf.layers.batch_normalization(bias4,training=True,reuse=tf.AUTO_REUSE) 
            bn3 = tf.contrib.layers.batch_norm(inputs=bias3,decay=0.9,updates_collections=None,is_training = self.is_training)
            self.conv3 = tf.nn.relu(bn3, name=scope)
            

        with tf.name_scope("conv4") as scope:
            kernel4 = tf.Variable(tf.truncated_normal([self.filter_size,self.filter_size, self.filter_num_2,self.filter_num_2], mean=0, stddev=0.1,
                                                      dtype=tf.float32), trainable=True,name="weights")
            conv = tf.nn.conv2d(self.conv3, kernel4, [1,1,1,1], padding="SAME")
            biases = tf.Variable(tf.constant(0, shape=[self.filter_num_2], dtype=tf.float32), trainable=True, name="biases")
            bias4 = tf.nn.bias_add(conv, biases)
            # bn6 = tf.layers.batch_normalization(bias6,training=True,reuse=tf.AUTO_REUSE) 
            bn4 = tf.contrib.layers.batch_norm(inputs=bias4,decay=0.9,updates_collections=None,is_training = self.is_training)
            self.conv4 = tf.nn.relu(bn4, name=scope)
            self.max_pool2=tf.nn.max_pool(self.conv4,ksize=[1,self.max_pool_size,self.max_pool_size,1],strides=[1,2,2,1],padding="VALID",name="pool2")
            activation2 = tf.nn.relu(self.max_pool2, name='activation2')
            

        with tf.name_scope("fc1") as scope:
            flatten = tf.reshape(activation2, (-1, activation2.shape[-1]*activation2.shape[-2]*activation2.shape[-3]))
            self.fc_weight1 = tf.Variable(tf.truncated_normal([tf.cast(flatten.shape[-1],tf.int32), self.fc_num_1], mean=0, stddev=0.1, dtype=tf.float32),trainable=True,name="fc_weight1")
            fc_bias1 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[self.fc_num_1]), trainable=True, name="fc_bias1")
            fc1 = tf.add(tf.matmul(flatten, self.fc_weight1), fc_bias1,name=scope)
            activation3 = tf.nn.relu(fc1, name='activation3')
            drop_out=tf.nn.dropout(activation3,self.keep_prob)

        with tf.name_scope("fc2") as scope:
            fc_weight2 = tf.Variable(tf.truncated_normal([self.fc_num_1, self.fc_num_2], mean=0, stddev=0.1, dtype=tf.float32),trainable=True,name="fc_weight2")
            fc_bias2 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[self.fc_num_2]), trainable=True, name="fc_bias2")
            fc2 = tf.add(tf.matmul(drop_out, fc_weight2), fc_bias2,name=scope)
            self.activation4 = tf.nn.relu(fc2, name='activation6')

        with tf.name_scope("softmax") as scope:
            softmax_weight = tf.Variable(tf.truncated_normal([self.fc_num_2, self.class_num], mean=0, stddev=0.1, dtype=tf.float32),
                                      trainable=True, name="softmax_weight")
            softmax_bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[self.class_num]), trainable=True, name="softmax_bias")
            self.logit = tf.add(tf.matmul(tf.cast(self.activation4,tf.float32), softmax_weight) , softmax_bias,name=scope)
            self.softmax =tf.nn.softmax(self.logit,name='softmax') #tf.nn.sigmoid(self.logit,name='sigmoid')#
            self.predict=tf.cast(tf.argmax(self.softmax, 1),tf.int32)


    def train(self,train_x,train_y,val_x,val_y):
        #self.loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(self.label,tf.int32),logits=self.logit)#+tf.contrib.layers.l2_regularizer(self.weight_decay)(self.fc_weight1)
        self.loss = focal_loss(y_true=tf.one_hot(tf.cast(self.label,tf.int32),2,1,0,-1),y_pred = self.logit, gamma=2,alpha=0.75)#+tf.contrib.layers.l2_regularizer(self.weight_decay)(self.fc_weight1)
        # self.loss = balanced_loss(y_true=tf.one_hot(tf.cast(self.label,tf.int32),2,1,0,-1),y_pred = self.logit)
        
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.predict,tf.float32), self.label), tf.float32))
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)            
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)        
   
        self.init_op = tf.global_variables_initializer()
        self.init_local_op = tf.local_variables_initializer()        
        
        var_list = tf.global_variables()
        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=5)

        with self.session as sess:
            sess.run(self.init_op)
            sess.run(self.init_local_op)
        
   
            for epoch in range(1,self.epochs+1):
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
                                                                                                 self.keep_prob:0.5,
                                                                                                 self.is_training:True})
                    # print(logit)
                    train_loss+=loss*len(train_y[i * self.batch_size:(i + 1) * self.batch_size])
                    train_predict = np.concatenate((train_predict, predict))
                    train_logit = np.concatenate((train_logit,logit[:,1]))

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
                    loss_,predict_,logit_=sess.run([self.loss,self.predict,self.softmax],feed_dict={self.input:val_x[j * self.predict_batch_size:(j + 1) * self.predict_batch_size],
                                                                self.label:val_y[j * self.predict_batch_size:(j + 1) * self.predict_batch_size],
                                                                                self.keep_prob:0.5,
                                                                                self.is_training:False})
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
 
            # Save model weights to disk
            save_path = self.saver.save(sess, model_path)
            print("Model saved in file: %s" % save_path)




if __name__=="__main__":
    sys.stdout = Logger("result_loss.txt" ) #将模型训练过程和结果保存到文件
    
    train_x=np.load("./data_patches/train_x_step_2.npy")
    train_y=np.load("./data_patches/train_y_step_2.npy")
    print('>>>train_x shape:',train_x.shape)
    print('>>>train_y shape:', train_y.shape)
    print('>>>positive instances:',np.sum(train_y==1),' negative instances:',np.sum(train_y==0))
    print()

    
    val_x=np.load("./data_patches/val_x_step_1.npy")
    val_y=np.load("./data_patches/val_y_step_1.npy")
    print('>>>val_x shape:',val_x.shape)
    print('>>>val_y shape:', val_y.shape)
    print('>>>positive instances:', np.sum(val_y == 1), ' negative instances:', np.sum(val_y == 0))
    print()
    
  


    model=FaultDetection()
    model.construct_network()
    model.train(train_x=train_x,train_y= train_y.astype(np.float32),val_x=val_x,val_y=val_y.astype(np.float32))



