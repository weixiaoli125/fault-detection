# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 22:05:55 2019

@author: DELL
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops

def  balanced_loss(y_true,y_pred):
    #输入：y_true：onehot编码的真实标签；y_pred:未经softmax的logits
    # your class weights
    flag = tf.constant(0.0,tf.float32)    
    neg = np.sum(y_true[:,0])
    pos = np.sum(y_true[:,1])
    if neg==0 or pos==0:
        y_true = tf.cast(y_true, tf.float32)
        # # compute your (unweighted) softmax cross entropy loss
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = y_pred)
        # # reduce the result to get your final loss
        loss = tf.reduce_mean(unweighted_losses)    
    else:
        y_true = tf.cast(y_true, tf.float32)
        neg = tf.reduce_sum(y_true[:,0])
        pos = tf.reduce_sum(y_true[:,1])
        wei1 = tf.expand_dims(tf.cast(pos/(neg+pos),tf.float32),0)
        wei2 =tf.expand_dims(tf.cast(neg/(neg+pos),tf.float32),0)
        class_weights = tf.expand_dims(tf.concat([wei1,wei2],axis=0),0)
        # deduce weights for batch samples based on their true label
        weights = tf.reduce_sum(class_weights * y_true, axis=1)
        # # compute your (unweighted) softmax cross entropy loss
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = y_pred)
        # # apply the weights, relying on broadcasting of the multiplication
        weighted_losses = unweighted_losses * weights
        # # reduce the result to get your final loss
        loss = tf.reduce_mean(weighted_losses)
    return loss


def focal_loss(y_true,y_pred,gamma,alpha):
    #输入：y_true：onehot编码的真实标签；y_pred:未经softmax的logits
    gamma = tf.cast(gamma, tf.float32)
    alpha = tf.cast(alpha, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    softmax = tf.nn.softmax(y_pred)
    epsilon = 1e-9    
    # your class weights
    wei1 = tf.expand_dims(1-alpha,0)
    wei2 =tf.expand_dims(alpha,0)
    class_weights = tf.expand_dims(tf.concat([wei1,wei2],axis=0),0)
    # deduce weights for batch samples based on their true label
    alpha_matrix = tf.reduce_sum(class_weights * y_true, axis=1)
    #calculate p_t
    p_t = y_true * (softmax+epsilon) + (tf.ones_like(y_true) - y_true+epsilon) * (tf.ones_like(y_true) - softmax+epsilon)
    ## 然后通过p_t和gamma得到weight
    weight = tf.pow((tf.ones_like(y_true) - p_t), gamma)
    ans = tf.reduce_sum(weight * tf.log(p_t),axis=1)
    loss = -alpha_matrix*ans
    return tf.reduce_mean(loss)



