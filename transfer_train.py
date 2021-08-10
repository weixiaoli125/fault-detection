# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 21:36:12 2021

@author: win10
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 00:22:42 2021

@author: win10
"""

import numpy as np
import tensorflow as tf
import os
import math
import datetime,time
import matplotlib.pyplot as plt
from focal_loss import focal_loss,balanced_loss,focal_loss_wei

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
old_path = "./case2/focalloss/"
path_transfer ="./case2/focalloss/transfer/"
def bottleneck(old_path,train_x,batch_size=1024):   
    log_dir = old_path + "log"
    model_path =old_path + "log/521model.ckpt.meta"
    index_path = old_path +"log/"
    saver = tf.train.import_meta_graph(model_path)# 加载图结构
    gragh = tf.get_default_graph()# 获取当前图，为了后续训练时恢复变量
    tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]# 得到当前图中所有变量的名称
    x = gragh.get_tensor_by_name('Placeholder:0')# 获取输入变量（占位符，由于保存时未定义名称，tf自动赋名称“Placeholder”）
    is_training = gragh.get_tensor_by_name('is_training:0')
    
    conv4 = gragh.get_tensor_by_name('conv4/activation2:0')# 获取bottleneck输出特征
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(index_path))
        train_features = np.ones((1,11,11,50))
        for i in range(0, int(math.ceil(train_x.shape[0] / batch_size))):        
            features =sess.run([conv4],feed_dict={x:train_x[i * batch_size:(i + 1) * batch_size],is_training:False})
            train_features = np.concatenate((train_features,np.array(features).squeeze()),axis=0)
    return train_features[1:,:,:,:]                                                                                     



class transfer_classifier:

    def __init__(self):

        self.graph = tf.reset_default_graph()
        
        self.width=11
        self.height=11
        self.channel=50
        self.fc_num_1= 32
        self.fc_num_2= 16
        self.class_num= 2

        self.epochs=60
        self.batch_size=1024
        self.learning_rate=0.0001
        
        self.beta=0.5       

    def construct_network(self):
        self.input=tf.placeholder(dtype=tf.float32,shape=(None,self.width,self.height,self.channel))
        self.label=tf.placeholder(dtype=tf.float32,shape=(None))
        self.keep_prob = tf.placeholder("float")
        
        with tf.name_scope("fc1") as scope:
            flatten = tf.reshape(self.input, (-1, self.input.shape[-1]*self.input.shape[-2]*self.input.shape[-3]))
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
            softmax_weight = tf.Variable(tf.truncated_normal([self.fc_num_2, self.class_num], mean=0, stddev=0.1, dtype=tf.float32),
                                      trainable=True, name="softmax_weight")
            softmax_bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[self.class_num]), trainable=True, name="softmax_bias")
            self.logit = tf.add(tf.matmul(tf.cast(self.activation6,tf.float32), softmax_weight) , softmax_bias,name=scope)

            self.softmax =tf.nn.softmax(self.logit,name='softmax') #tf.nn.sigmoid(self.logit,name='sigmoid')#
            self.predict=tf.cast(tf.argmax(self.softmax, 1),tf.int32)

    def train(self,train_x,train_y,val_x,val_y,old_path):
        # self.loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(self.label,tf.int32),logits=self.logit)#+tf.contrib.layers.l2_regularizer(self.weight_decay)(self.fc_weight1)
        self.loss = focal_loss(y_true=tf.one_hot(tf.cast(self.label,tf.int32),2,1,0,-1),y_pred = self.logit, gamma=2,alpha=0.85)#+tf.contrib.layers.l2_regularizer(self.weight_decay)(self.fc_weight1)
        # self.loss = balanced_loss(y_true=tf.one_hot(tf.cast(self.label,tf.int32),2,1,0,-1),y_pred=self.logit)#96
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)        
   
        self.init_op = tf.global_variables_initializer()
        self.init_local_op = tf.local_variables_initializer()        

        # var_list = tf.global_variables()
        # self.saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
        ref_meta_path = old_path + "log/521model.ckpt.meta" # 后缀是'.ckpt.meta'的文件
        ref_graph_weight = old_path +"log/" # 后缀是'.ckpt'的文件，里面是各个tensor的值
        # ref_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc1') # 既然有这个scope，其实第1个方法中，导入graph后，可以不用返回的vgg_saver，再新建一个指定var_list的vgg_saver就好了，恩，需要传入一个var_list的参数
        restore_var = [v for v in tf.global_variables()]
        variables_to_resotre = [v for v in restore_var if v.name.split('/')[0] == 'fc1' or 
                                v.name.split('/')[0] == 'fc2' or 
                                v.name.split('/')[0] == 'softmax']
        for i in variables_to_resotre:
            print(i)
            
        ckpt = tf.train.get_checkpoint_state(ref_graph_weight)
        saver_ref = tf.train.Saver(var_list=variables_to_resotre)

              
        saver = tf.train.Saver(max_to_keep=5) # 这个是当前新图的saver
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        with tf.Session(config=config) as sess:
            sess.run(self.init_op)
            sess.run(self.init_local_op)
            saver_ref.restore(sess, ckpt.model_checkpoint_path)#使用导入图的saver来恢复
        
            starttime = datetime.datetime.now()
            for epoch in range(1,self.epochs+1):
                np.random.seed(epoch)
                np.random.shuffle(train_x)
                np.random.seed(epoch)
                np.random.shuffle(train_y)

                
                train_loss=0
                train_predict = np.array([])
                train_logit = np.array([])
                for i in range(0, int(math.ceil(train_x.shape[0] / self.batch_size))):
                    _,loss,predict,logit =sess.run([self.optimizer,self.loss,self.predict,self.softmax],feed_dict={self.input:train_x[i * self.batch_size:(i + 1) * self.batch_size],
                                                                                        self.label:train_y[i * self.batch_size:(i + 1) * self.batch_size],
                                                                                                  self.keep_prob:0.5})
                    # print(logit)
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
                
                val_predict=np.array([])
                val_logit=np.array([])
                val_loss=0
                #test_predict=[]
                for j in range(0, int(math.ceil(val_x.shape[0]/self.batch_size))):
                    loss_,predict_,logit_=sess.run([self.loss,self.predict,self.softmax],feed_dict={self.input:val_x[j * self.batch_size:(j + 1) * self.batch_size],
                                                                self.label:val_y[j * self.batch_size:(j + 1) * self.batch_size],
                                                                                self.keep_prob:0.5})
                    val_loss+=loss_*len(val_y[j * self.batch_size:(j + 1) * self.batch_size])
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
            save_path = saver.save(sess, model_path_transfer)
            print("Model saved in file: %s" % save_path)
            endtime = datetime.datetime.now()
            print('>>>test time:',(endtime - starttime).seconds)    


transfer_x=np.load('./data_patches/train_x_transfer_3.npy')
train_x = transfer_x[:int(0.9*transfer_x.shape[0]),:,:,:]
val_x = transfer_x[int(0.9*transfer_x.shape[0]):,:,:,:]

features_train = bottleneck(old_path,train_x)
features_val = bottleneck(old_path,val_x)

transfer_y=np.load('./data_patches/train_y_transfer_3.npy')
label_train = transfer_y[:int(0.9*transfer_y.shape[0]),]
label_val = transfer_y[int(0.9*transfer_y.shape[0]):,]


print('>>>features_train shape:',features_train.shape)
print('>>>label_train shape:', label_train.shape)
print('>>>positive instances:', np.sum(label_train == 1), ' negative instances:', np.sum(label_train == 0))
print()

print('>>>features_val shape:',features_val.shape)
print('>>>label_val shape:', label_val.shape)
print('>>>positive instances:', np.sum(label_val == 1), ' negative instances:', np.sum(label_val == 0))
print()

losses = {'train':[],'val':[]}  
model_path_transfer = path_transfer+"log/521model.ckpt"

model=transfer_classifier()
model.construct_network()
model.train(train_x=features_train,train_y= label_train.astype(np.float32),
            val_x=features_val,val_y= label_val.astype(np.float32),old_path=old_path)

plt.figure(1)
plt.plot(losses['train'], label='Training loss')
plt.legend()
_ = plt.ylim()
plt.savefig(path_transfer+'log/figure_train_loss.png')
plt.plot(losses['val'], label='val loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
_ = plt.ylim()
plt.savefig(path_transfer+'log/figure_val_loss.png')

