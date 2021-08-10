# fault detection: Seismic Fault Detection Using Convolutional Neural Networks with Focal Loss
Xiao-Li We, Chun-Xia Zhang, Sang-Woon Kim, Kai-Li Jing, Zhuang-Zhuang Xie, Shuang Xu

**This code implemented by Xiao-Li We for 2D seismic fault detection is based on TensorFlow 1.14.0. GPU is RTX 2080 Ti.

## Getting Start for fault detection

##Run an example for fault prediction

If you would just like to try predict example, then you can run test_transfer_matrix.py and dowmload the folder[log] to use.

### Dataset
**To train our CNN network, with the help of the open source code from Hale ( https://github.com/dhale/ipf), we automatically created 800 pairs of synthetic seismic image and corresponding binary masks. 

All images are composed of 200 by 200 pixels. And then we extract patches from them. 
The patch training set (train_x_step_3,train_y_step_3; train_x_step_2,train_y_step_2) and validition set (val_x_step_1,val_y_step_1)  can be generated by proprecessing.py. 
And patch test set (train_x_transfer_3,train_y_transfer_3;test_x_transfer_3,test_y_transfer_3)can be extracted by preprocessing_transfer.py

#Two models in folders case1 and case2 are FCFT-Focal loss-Case1 and FCFT-Focal loss-Case2. 

#training
Run pre_train.py to start the pre_traininging process.
Run transfer_train.py to start the transfer learning stage.


## Validation on field examples

To verify the generalization ability of our model, we also test it with some real data from Netherland offshore F3 block complete.(https://www.opendtect.org/osr/Main/NetherlandsOffshoreF3BlockComplete4GB).

More detail can be found in manuscript.
