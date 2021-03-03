# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 21:38:56 2020

@author: win10
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io,data
import cv2
img = cv2.imread("G:/wei_loss/datame/results/focal_loss/no_padding/test_pred/image/test_pred_722.png")
cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = np.ones((3, 3), np.uint8)
#腐蚀操作
erosion1  = cv2.erode(img, kernel, iterations=1)
erosion2  = cv2.erode(img, kernel, iterations=2)
erosion3  = cv2.erode(img, kernel, iterations=3)
erosion4  = cv2.erode(img, kernel, iterations=4)
erosion5  = cv2.erode(img, kernel, iterations=5)
erosion6  = cv2.erode(img, kernel, iterations=6)
erosion7  = cv2.erode(img, kernel, iterations=7)
erosion8  = cv2.erode(img, kernel, iterations=8)
res = np.hstack((erosion1, erosion2, erosion3,erosion4, erosion5, erosion6, erosion7, erosion8))
cv2.imshow('res', res)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('G:/wei_loss/datame/results/focal_loss/no_padding/test_pred/erosion_722.png',erosion6)

# 膨胀操作(不咋行)
# kernel = np.ones((3, 3), np.uint8)

# dilate1  = cv2.dilate(erosion6, kernel, iterations=1)
# dilate2  = cv2.dilate(erosion6, kernel, iterations=2)
# dilate3  = cv2.dilate(erosion6, kernel, iterations=3)
# dilate4  = cv2.dilate(erosion6, kernel, iterations=4)
# dilate5  = cv2.dilate(erosion6, kernel, iterations=5)
# dilate6  = cv2.dilate(erosion6, kernel, iterations=6)

# res = np.hstack((dilate1, dilate2, dilate3,dilate4,dilate5,dilate6))
# cv2.imshow('res', res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#开运算：先腐蚀后膨胀,type=cv2.MORPH_OPEN
kernel = np.ones((3, 3), np.uint8)
open1 = cv2.morphologyEx(erosion6, cv2.MORPH_OPEN, kernel, iterations=1)
open2 = cv2.morphologyEx(erosion6, cv2.MORPH_OPEN, kernel, iterations=2)
open3 = cv2.morphologyEx(erosion6, cv2.MORPH_OPEN, kernel, iterations=3)

res1 = np.hstack((open1, open2, open3))
cv2.imshow('res', res1)
cv2.waitKey(0)
cv2.destroyAllWindows()

#闭运算：先膨胀后腐蚀,type=cv2.MORPH_CLOSED
kernel = np.ones((3, 3), np.uint8)
closed1 = cv2.morphologyEx(erosion6, cv2.MORPH_CLOSE, kernel, iterations=1)
closed2 = cv2.morphologyEx(erosion6, cv2.MORPH_CLOSE, kernel, iterations=2)
closed3 = cv2.morphologyEx(erosion6, cv2.MORPH_CLOSE, kernel, iterations=3)
closed4 = cv2.morphologyEx(erosion6, cv2.MORPH_CLOSE, kernel, iterations=4)
closed5 = cv2.morphologyEx(erosion6, cv2.MORPH_CLOSE, kernel, iterations=5)
closed6 = cv2.morphologyEx(erosion6, cv2.MORPH_CLOSE, kernel, iterations=6)

res2 = np.hstack((closed1, closed2, closed3,closed4, closed5, closed6))
cv2.imshow('res', res2)

#细化操作

def VThin(image,array):
    h = image.shape[0]
    w = image.shape[1]
    NEXT = 1
    for i in range(h):
        for j in range(w):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i,j-1]+image[i,j]+image[i,j+1] if 0<j<w-1 else 1
                if image[i,j] == 0  and M != 0:                  
                    a = [0]*9
                    for k in range(3):
                        for l in range(3):
                            if -1<(i-1+k)<h and -1<(j-1+l)<w and image[i-1+k,j-1+l]==255:
                                a[k*3+l] = 1
                    sum = a[0]*1+a[1]*2+a[2]*4+a[3]*8+a[5]*16+a[6]*32+a[7]*64+a[8]*128
                    image[i,j] = array[sum]*255
                    if array[sum] == 1:
                        NEXT = 0
    return image
    
def HThin(image,array):
    h = image.shape[0]
    w = image.shape[1]
    NEXT = 1
    for j in range(w):
        for i in range(h):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i-1,j]+image[i,j]+image[i+1,j] if 0<i<h-1 else 1   
                if image[i,j] == 0 and M != 0:                  
                    a = [0]*9
                    for k in range(3):
                        for l in range(3):
                            if -1<(i-1+k)<h and -1<(j-1+l)<w and image[i-1+k,j-1+l]==255:
                                a[k*3+l] = 1
                    sum = a[0]*1+a[1]*2+a[2]*4+a[3]*8+a[5]*16+a[6]*32+a[7]*64+a[8]*128
                    image[i,j] = array[sum]*255
                    if array[sum] == 1:
                        NEXT = 0
    return image
    
def Xihua(image,array,num=10):
    iXihua=image.copy()
    for i in range(num):
        VThin(iXihua,array)
        HThin(iXihua,array)
    return iXihua

# def Two(image):
#     w = image.shape[0]
#     h = image.shape[1]
#     size = (w,h)
#     iTwo =  np.zeros(size, np.uint8)
#     for i in range(w):
#         for j in range(h):
#             iTwo[i,j] = 0 if image[i,j] < 150 else 255
#     return iTwo


array = [0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
         1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,1,\
         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,0,\
         1,1,0,0,1,1,1,0,1,1,0,0,1,0,0,0]

import matplotlib.pyplot as plt        
image1 = cv2.imread("G:/wei_loss/datame/pred_test_step_1/image/test_pred_740.png",cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("C:/Users/win10/Desktop/wei1.png",cv2.IMREAD_GRAYSCALE)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
# eroded = cv2.erode(image,kernel) 
# # dilated = cv2.dilate(eroded,kernel)      #膨胀图像         #红色阈值图像
# cv2.imshow("Eroded Image",eroded)           #显示腐蚀后的图像
# # cv2.imshow("Dilated Image",dilated)   

# iTwo = Two(erosion6)
iThin = Xihua(img,array)
cv2.imshow('image',img)
cv2.imshow('iTwo',iTwo)
cv2.imshow('iThin',iThin)




#细化算法
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Zhang Suen thining algorythm
def Zhang_Suen_thining(img):
    # get shape
    H, W, C = img.shape

    # prepare out image
    out = np.zeros((H, W), dtype=np.int)
    out[img[..., 0] > 0] = 1

    # inverse
    out = 1 - out

    while True:
        s1 = []
        s2 = []

        # step 1 ( rasta scan )
        for y in range(1, H-1):
            for x in range(1, W-1):
                
                # condition 1
                if out[y, x] > 0:
                    continue

                # condition 2
                f1 = 0
                if (out[y-1, x+1] - out[y-1, x]) == 1:
                    f1 += 1
                if (out[y, x+1] - out[y-1, x+1]) == 1:
                    f1 += 1
                if (out[y+1, x+1] - out[y, x+1]) == 1:
                    f1 += 1
                if (out[y+1, x] - out[y+1,x+1]) == 1:
                    f1 += 1
                if (out[y+1, x-1] - out[y+1, x]) == 1:
                    f1 += 1
                if (out[y, x-1] - out[y+1, x-1]) == 1:
                    f1 += 1
                if (out[y-1, x-1] - out[y, x-1]) == 1:
                    f1 += 1
                if (out[y-1, x] - out[y-1, x-1]) == 1:
                    f1 += 1

                if f1 != 1:
                    continue
                    
                # condition 3
                f2 = np.sum(out[y-1:y+2, x-1:x+2])
                if f2 < 2 or f2 > 6:
                    continue
                
                # condition 4
                # x2 x4 x6
                if (out[y-1, x] + out[y, x+1] + out[y+1, x]) < 1 :
                    continue

                # condition 5
                # x4 x6 x8
                if (out[y, x+1] + out[y+1, x] + out[y, x-1]) < 1 :
                    continue
                    
                s1.append([y, x])

        for v in s1:
            out[v[0], v[1]] = 1

        # step 2 ( rasta scan )
        for y in range(1, H-1):
            for x in range(1, W-1):
                
                # condition 1
                if out[y, x] > 0:
                    continue

                # condition 2
                f1 = 0
                if (out[y-1, x+1] - out[y-1, x]) == 1:
                    f1 += 1
                if (out[y, x+1] - out[y-1, x+1]) == 1:
                    f1 += 1
                if (out[y+1, x+1] - out[y, x+1]) == 1:
                    f1 += 1
                if (out[y+1, x] - out[y+1,x+1]) == 1:
                    f1 += 1
                if (out[y+1, x-1] - out[y+1, x]) == 1:
                    f1 += 1
                if (out[y, x-1] - out[y+1, x-1]) == 1:
                    f1 += 1
                if (out[y-1, x-1] - out[y, x-1]) == 1:
                    f1 += 1
                if (out[y-1, x] - out[y-1, x-1]) == 1:
                    f1 += 1

                if f1 != 1:
                    continue
                    
                # condition 3
                f2 = np.sum(out[y-1:y+2, x-1:x+2])
                if f2 < 2 or f2 > 6:
                    continue
                
                # condition 4
                # x2 x4 x8
                if (out[y-1, x] + out[y, x+1] + out[y, x-1]) < 1 :
                    continue

                # condition 5
                # x2 x6 x8
                if (out[y-1, x] + out[y+1, x] + out[y, x-1]) < 1 :
                    continue
                    
                s2.append([y, x])

        for v in s2:
            out[v[0], v[1]] = 1

        # if not any pixel is changed
        if len(s1) < 1 and len(s2) < 1:
            break

    out = 1 - out
    out = out.astype(np.uint8) * 255

    return out


# Read image
erosion = cv2.imread("G:/wei_loss/datame/results/focal_loss/no_padding/test_pred/erosion_722.png").astype(np.float32)

# Zhang Suen thining
out = Zhang_Suen_thining(erosion)

cv2.imwrite('G:/wei_loss/datame/results/focal_loss/no_padding/test_pred/thin_722.png',out)
# Save result
cv2.imwrite("out.png", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

#-----------------------------直线检测-------------------------------------
#img = cv2.imread("C:/AE_PUFF/python_vision/2018_04_27/kk-3.jpg")
img = cv2.imread('G:/wei_loss/datame/results/focal_loss/no_padding/test_pred/thin_722.png')
cv2.imshow('origin_img', img)
height = img.shape[0]  # 高度
width  = img.shape[1]  # 宽度
cut_img = img
 
gray = cv2.cvtColor(cut_img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray_img', gray)
cv2.waitKey(0)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
 
lines = cv2.HoughLines(edges, 1, np.pi/180, 118)
result = cut_img.copy()
minLineLength = 30 # height/32
maxLineGap = 10 # height/40
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength, maxLineGap)
 
for x1, y1, x2, y2 in lines[0]:
    cv2.line(result, (x1, y1), (x2, y2), (0,255,0), 2)
 
cv2.imshow('result', result)


import cv2 as cv
import numpy as np

#标准霍夫线变换
def line_detection(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)  #apertureSize参数默认其实就是3
    cv.imshow("edges", edges)
    lines = cv.HoughLines(edges, 1, np.pi/180, 10)
    for line in lines:
        rho, theta = line[0]  #line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的。
        a = np.cos(theta)   #theta是弧度
        b = np.sin(theta)
        x0 = a * rho    #代表x = r * cos（theta）
        y0 = b * rho    #代表y = r * sin（theta）
        x1 = int(x0 + 1000 * (-b)) #计算直线起点横坐标
        y1 = int(y0 + 1000 * a)    #计算起始起点纵坐标
        x2 = int(x0 - 1000 * (-b)) #计算直线终点横坐标
        y2 = int(y0 - 1000 * a)    #计算直线终点纵坐标    注：这里的数值1000给出了画出的线段长度范围大小，数值越小，画出的线段越短，数值越大，画出的线段越长
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)    #点的坐标必须是元组，不能是列表。
    cv.imshow("image-lines", image)

#统计概率霍夫线变换
def line_detect_possible_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)  # apertureSize参数默认其实就是3
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, 5, minLineLength=10, maxLineGap=5)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imshow("line_detect_possible_demo",image)

src = cv.imread('G:/wei_loss/datame/results/focal_loss/no_padding/test_pred/thin_722.png')
print(src.shape)
cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE) 
cv.imshow('input_image', src)
line_detection(src)
src = cv.imread('G:/wei_loss/datame/results/focal_loss/no_padding/test_pred/thin_722.png') #调用上一个函数后，会把传入的src数组改变，所以调用下一个函数时，要重新读取图片
line_detect_possible_demo(src)