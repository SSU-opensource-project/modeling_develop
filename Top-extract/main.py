# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 23:28:33 2019

@author: Wei-Hsiang, Shen
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from cloth_detection import Detect_Clothes_and_Crop
from utils_my import Read_Img_2_Tensor, Save_Image, Load_DeepFashion2_Yolov3

model = Load_DeepFashion2_Yolov3()

img_path = './images/Mar_2.jpg'

# Read image
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = Read_Img_2_Tensor(img_path)

# Clothes detection and crop the image
img_crop = Detect_Clothes_and_Crop(img_tensor, model)

# Transform the image to gray_scale
cloth_img = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)

plt.imshow(img)
plt.title('Input image')
plt.show()

plt.imshow(img_crop)
plt.title('Cloth detection and crop')
plt.show()
# img_resize = img_crop.resize((512, 512))  # Resize를 할지ㅣ말지..
Save_Image(img_crop, './images/Mar_2_2.jpg')

#img = Image.open('./images/test12345_1_crop.jpg')

#img_resize = img.resize((512, 512))
# img_resize.save('./images/Mar_1_1.jpg')

# img_resize_lanczos = img.resize((256, 256), Image.LANCZOS)
