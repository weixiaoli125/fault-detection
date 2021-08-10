# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 20:33:06 2021

@author: win10
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 17:05:41 2021

@author: win10
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 20:22:51 2021

@author: win10
"""
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:10:02 2020

@author: lyric
"""

import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import math
import datetime,time
import matplotlib.pyplot as plt


from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from skimage import io,data

def bottleneck(old_path,test_x):
    tf.reset_default_graph()
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
        features =sess.run([conv4],feed_dict={x:test_x,is_training:False})           
        features = np.array(features).squeeze(0)
    return features  


def transfer_test(transfer_path,test_features,dropout_rate):
    tf.reset_default_graph()
    log_dir = transfer_path + "log"
    model_path = transfer_path + "log/521model.ckpt.meta"
    index_path = transfer_path +"log/"
    saver = tf.train.import_meta_graph(model_path)# 加载图结构
    gragh = tf.get_default_graph()# 获取当前图，为了后续训练时恢复变量
    tensor_name_list = [tensor.name for tensor in gragh.as_graph_def().node]# 得到当前图中所有变量的名称
    x = gragh.get_tensor_by_name('Placeholder:0')# 获取输入变量（占位符，由于保存时未定义名称，tf自动赋名称“Placeholder”）
    y = gragh.get_tensor_by_name('Placeholder_1:0')
    keep_prob = gragh.get_tensor_by_name('Placeholder_2:0')
    
    softmax = gragh.get_tensor_by_name('softmax/softmax:0')
    pred = gragh.get_tensor_by_name('softmax/Cast:0')# 获取网络输出值   with tf.Session() as sess:
        
    with tf.Session() as sess:    
        saver.restore(sess, tf.train.latest_checkpoint(index_path))        
        predict_batch,logit_batch = sess.run([pred,softmax],
                                             feed_dict={x:test_features,
                                                        keep_prob:dropout_rate})
    return np.array(predict_batch),np.array(logit_batch)
        


if __name__ == '__main__':      
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


    old_path = "./case2/focalloss/"
    transfer_path ="./case2/focalloss/transfer/"
    predict_batch_size = 1024#67006
    dropout_rate = 1.0
    test_x=np.load(r".\data_patches\test_x_transfer_3.npy")
    test_y = np.load(r".\data_patches\test_y_transfer_3.npy")
    print('>>>test_x shape:',test_x.shape)
    print('>>>test_y shape:', test_y.shape)
    print('>>>positive instances:', np.sum(test_y == 1), ' negative instances:', np.sum(test_y == 0))
    print()

    test_predict = np.array([])
    test_logit = np.array([])
    logits = np.transpose(np.array([[],[]]))    
    starttime = datetime.datetime.now()
    for i in range(0, int(math.ceil(test_x.shape[0] /predict_batch_size))):
        print(i)
        test_features = bottleneck(old_path,test_x[i * predict_batch_size:(i + 1) * predict_batch_size])
        predict_batch,logit_batch = transfer_test(transfer_path,test_features,dropout_rate)
        test_predict = np.concatenate((test_predict, predict_batch))
        test_logit = np.concatenate((test_logit,logit_batch[:,1]))
        logits = np.concatenate((logits,logit_batch),axis=0)

    test_accuracy=np.sum(test_predict==test_y)/(len(test_y)+1e-9)
    test_precision=np.sum((test_y==1)*(test_predict==1))/(np.sum(test_predict==1)+1e-9)
    test_recall=np.sum((test_y==1)*(test_predict==1))/(np.sum(test_y==1)+1e-9)
    test_f1_score=2*test_precision*test_recall/(test_precision+test_recall+1e-9)
    test_specificity = np.sum((test_y==0)*(test_predict==0))/(np.sum(test_y==0)+1e-9)

    print(' test_accuracy:{:.2%}'.format(test_accuracy),' test_precision:',test_precision,
          ' test_recall:',test_recall,' test_f1_score:',test_f1_score,
          'test_specificity:',test_specificity)     


    print('------------confusion_matrix_test:',confusion_matrix(test_y,test_predict))
    fpr, tpr, thresholds = roc_curve(np.array(test_y), np.array(test_logit), pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure(1)
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
              lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC')
    plt.legend(loc="lower right")
    
    plt.show()
    print('------------auc_test:',roc_auc)
    plt.savefig(transfer_path+"test_pred_3/roc.jpeg",bbox_inches="tight", pad_inches=0.0)
           
    
    endtime = datetime.datetime.now()
    print('>>>test time:',(endtime - starttime).seconds)       

