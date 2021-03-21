# fault detection: Seismic Fault Detection Using Convolutional Neural Networks with Focal Loss
Xiao-Li We, Chun-Xia Zhang, Sang-Woon Kim, Zhuang-Zhuang Xie, Shuang Xu

**This code implemented by Xiao-Li We for 2D seismic fault detection is based on TensorFlow 1.14.0 and Intel Core i9- 9900H CPU@2.60 GHz. GPU is RTX 2080 Ti.

## Getting Start for fault detection

##Run an example for fault prediction
If you would just like to try predict example, then you can dowmload the folder[log] to use.

### Dataset
**To train our CNN network, with the help of the open source code from Hale ( https://github.com/dhale/ipf), we automatically created 800 pairs of synthetic seismic image and corresponding binary masks. All images are composed of 200 by 200 pixels. And then we extract patches from them. 
**The training and validation patches can be downloaded in  folder [patches set].

#training
Run trainmain.py to start training a new model 

## Test on a synthetic example
test.py computed the classification results by our model

## Validation on field examples
To verify the generalization ability of our model, we also test it with some real data from Netherland offshore F3 block complete.(https://www.opendtect.org/osr/Main/NetherlandsOffshoreF3BlockComplete4GB).

More detail can be found in manuscript.
