# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 22:05:55 2019

@author: DELL
"""

import tensorflow as tf
from tensorflow.python.ops import array_ops

def focal_loss(y_pred,y_true,gamma,alpha):
    # y_true为one-hot
    gamma = tf.cast(gamma, tf.float32)
    alpha = tf.cast(alpha, tf.float32)
    epsilon = 1e-9
    #zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
    ones = array_ops.ones_like(y_pred, dtype=y_pred.dtype)
    p_t = tf.where(tf.cast(y_true,tf.float32) >= y_pred, tf.cast(y_pred + epsilon,tf.float32),tf.cast(ones,tf.float32)-tf.cast(y_pred,tf.float32)+epsilon)
    matrix_1 = tf.expand_dims(tf.cast(y_true[:,0],tf.float32)*alpha,0)
    matrix_2 = tf.expand_dims(tf.cast(y_true[:,1],tf.float32)*(1.0-alpha),0)
    alpha_matrix = tf.concat([matrix_1,matrix_2],0)
    alpha_matrix = tf.transpose(alpha_matrix,[1,0])
    loss_1 = tf.pow(ones-p_t,gamma)*tf.log(p_t)
    loss_2 = - alpha_matrix * loss_1 
    loss = tf.reduce_mean(tf.reduce_sum(loss_2,axis=1))
    return(loss)


from keras import backend as K
def focal_loss_fixed(y_true, y_pred,gamma,alpha): # y_true 是个一阶向量, 下式按照加号分为左右两部分 # 注意到 y_true的取值只能是 0或者1 (假设二分类问题)，可以视为“掩码” # 加号左边的 y_true*alpha 表示将 y_true中等于1的槽位置为标量 alpha # 加号右边的 (ones-y_true)*(1-alpha) 则是将等于0的槽位置为 1-alpha 
    y_true = tf.cast(y_true,tf.float32)
    ones = K.ones_like(y_true) 
    alpha_t = y_true*alpha + (ones-y_true)*(1-alpha) # 类似上面，y_true仍然视为 0/1 掩码 # 第1部分 `y_true*y_pred` 表示 将 y_true中为1的槽位置为 y_pred对应槽位的值 # 第2部分 `(ones-y_true)*(ones-y_pred)` 表示 将 y_true中为0的槽位置为 (1-y_pred)对应槽位的值 # 第3部分 K.epsilon() 避免后面 log(0) 溢出 
    p_t = y_true*y_pred + (ones-y_true)*(ones-y_pred) + K.epsilon() # 就是公式的字面意思 
    focal_loss = -alpha_t * K.pow((ones-p_t),gamma) * K.log(p_t) 
    return tf.reduce_mean(tf.reduce_sum(focal_loss,axis=1))


def balanced_loss(y_true,y_pred,alpha): # y_true 是个一阶向量, 下式按照加号分为左右两部分 # 注意到 y_true的取值只能是 0或者1 (假设二分类问题)，可以视为“掩码” # 加号左边的 y_true*alpha 表示将 y_true中等于1的槽位置为标量 alpha # 加号右边的 (ones-y_true)*(1-alpha) 则是将等于0的槽位置为 1-alpha 
    y_true = tf.cast(y_true,tf.float32)
    ones = K.ones_like(y_true) 
    alpha_t = y_true*alpha + (ones-y_true)*(1-alpha) # 类似上面，y_true仍然视为 0/1 掩码 # 第1部分 `y_true*y_pred` 表示 将 y_true中为1的槽位置为 y_pred对应槽位的值 # 第2部分 `(ones-y_true)*(ones-y_pred)` 表示 将 y_true中为0的槽位置为 (1-y_pred)对应槽位的值 # 第3部分 K.epsilon() 避免后面 log(0) 溢出 
    p_t = y_true*y_pred + (ones-y_true)*(ones-y_pred) + K.epsilon() # 就是公式的字面意思 
    balanced_loss = -alpha_t * K.log(p_t) 
    return tf.reduce_mean(tf.reduce_sum(balanced_loss,axis=1))
    
    
    






















