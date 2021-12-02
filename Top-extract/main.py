# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 23:28:33 2019

@author: Wei-Hsiang, Shen
"""

from utils_my import Read_Img_2_Tensor, Load_DeepFashion2_Yolov3, Draw_Bounding_Box
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tensorflow.python.ops.gen_image_ops import image_projective_transform_v3_eager_fallback
from cloth_detection import Detect_Clothes_and_Crop
from utils_my import Read_Img_2_Tensor, Save_Image, Load_DeepFashion2_Yolov3

model = Load_DeepFashion2_Yolov3()

# 긁어온거


def Detect_Clothes(img, model_yolov3, eager_execution=True):
    """Detect clothes in an image using Yolo-v3 model trained on DeepFashion2 dataset"""
    img = tf.image.resize(img, (416, 416))

    t1 = time.time()
    if eager_execution == True:
        boxes, scores, classes, nums = model_yolov3(img)
        # change eager tensor to numpy array
        boxes, scores, classes, nums = boxes.numpy(
        ), scores.numpy(), classes.numpy(), nums.numpy()
    else:
        boxes, scores, classes, nums = model_yolov3.predict(img)
    t2 = time.time()
    print('Yolo-v3 feed forward: {:.2f} sec'.format(t2 - t1))

    class_names = ['short_sleeve_top', 'long_sleeve_top', 'short_sleeve_outwear', 'long_sleeve_outwear',
                   'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeve_dress',
                   'long_sleeve_dress', 'vest_dress', 'sling_dress']

    # Parse tensor
    list_obj = []
    for i in range(nums[0]):
        obj = {'label': class_names[int(
            classes[0][i])], 'confidence': scores[0][i]}
        obj['x1'] = boxes[0][i][0]
        obj['y1'] = boxes[0][i][1]
        obj['x2'] = boxes[0][i][2]
        obj['y2'] = boxes[0][i][3]
        list_obj.append(obj)

    return list_obj


def Detect_Clothes_and_Crop(img_tensor, model, threshold=0.5):
    global img_crop
    global flag
    list_obj = Detect_Clothes(img_tensor, model)

    img = np.squeeze(img_tensor.numpy())
    img_width = img.shape[1]
    img_height = img.shape[0]

    class_names = ['short_sleeve_top', 'long_sleeve_top',
                   'vest', 'short_sleeve_outwear', 'long_sleeve_outwear', 'long_sleeve_dress', 'vest_dress', 'sling_dress', 'short_sleeve_dress']
    # , 'shorts', 'trousers', 'short_sleeve_dress',
    # ]  # 'long_sleeve_dress', 'vest_dress', 'sling_dress']
    # crop out one cloth
    flag = 0
    for obj in list_obj:
        if obj['label'] in class_names and obj['confidence'] > threshold:
            # if obj['label'] == 'short_sleeve_top' and obj['confidence']>threshold:
            img_crop = img[int(obj['y1']*img_height): int(obj['y2']*img_height),
                           int(obj['x1']*img_width): int(obj['x2']*img_width), :]
            flag = 1

    return img_crop


if __name__ == "__main__":
    tmpfile = Image.open('./tmp_strange2.jpg')  # Crop 안될시에 넣을 이상한 사진
    for i in range(1, 4950):  # 사진범위 조정
        if(i % 100 == 0):
            print(i)
        img_path = '../selenium/New_Many_Pics/'+str(i)+'.jpg'  # 이미지 위치
# Read image
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = Read_Img_2_Tensor(img_path)

# Clothes detection and crop the image
            img_crop = Detect_Clothes_and_Crop(img_tensor, model)

            if(flag == 0):
                tmpfile.save('../selenium/test_t/'+str(i)+'.jpg')
                flag = 1
            else:
                img_crop = cv2.resize(img_crop, dsize=(
                    220, 220), interpolation=cv2.INTER_LINEAR)
                Save_Image(
                    img_crop, '../selenium/test_t/'+str(i)+'.jpg')
        except Exception as e:
            print(i)
            tmpfile.save('../selenium/test_t/'+str(i)+'.jpg')
            pass
