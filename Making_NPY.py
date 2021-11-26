from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


class FeatureExtractor:
    def __init__(self):
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input,
                           outputs=base_model.get_layer('fc1').output)
        # self.features = np.array(
        #    np.load(os.getcwd() + "/Feature/Total_RealLast_erase.npy"))  # 이미지의 특징이 저장된 npy파일
        # self.Img_features = np.array(
        #    np.load(os.getcwd() + "/Feature/Image_3.npy"))  # 이미지 경로가 저장된 npy 파일/ 수정가능.수정필요. (그리고 굳이 안해도 되는것같음)

    def extract(self, img):
        img = img.resize((224, 224))
        img = img.convert('RGB')

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)


fe = FeatureExtractor()
img_paths = []
features = []

#img_paths = fe.Img_features
feature_path = "./Feature/Total_RealLast_Erase_1_2(NotCut).npy"
for i in range(1, 4950):  # 범위조정
    if i % 100 == 0:
        print(i)
    try:
        image_path = "./New_Many_Pics/"+str(i)+".jpg"  # 이미지 주소
        img_paths.append(image_path)

        #img_P1 = fe.Img_featurecs
        feature = fe.extract(img=Image.open(image_path))
        features.append(feature)  # features라는

#         #  # features라는 배열에 집어넣음
    #    np.save(feature_path, feature)
        np.save(feature_path, features)
    except Exception as e:
        print('예외가 발생했습니다.', e)

img_path_1 = "./Feature/Image_1_2((NotCut)).npy"
np.save(img_path_1, img_paths)

#feature_path = "./Feature/Total_RealLast_Erase.npy"
#np.save(feature_path, features)

"""
Ff = fe.features
img = Image.open("Result968.png")  # 궁금한 이미지
query = fe.extract(img)
img_paths = fe.Img_features

dists = np.linalg.norm(Ff - query, axis=1)
ids = np.argsort(dists)[:30]
print(ids)  # 사진 번호들 -> 문제 : 전처리 도중에 중간중간 번호가 없는 현상 발생.
scores = [(dists[id], img_paths[id], id) for id in ids]

axes = []
fig = plt.figure(figsize=(8, 8))
for a in range(5*6):
    score = scores[a]
    axes.append(fig.add_subplot(5, 6, a+1))
    subplot_title = str(round(score[0], 2)) + \
        "/m" + str(score[2]+1)  # dists + id
    axes[-1].set_title(subplot_title)
    plt.axis('off')
    plt.imshow(Image.open(score[1]))  # 이미지 창 열음
fig.tight_layout()
plt.show()
"""
