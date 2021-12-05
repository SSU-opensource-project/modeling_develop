from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

###


class FeatureExtractor:
    def __init__(self):
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input,
                           outputs=base_model.get_layer('fc1').output)
        self.features = np.array(
            np.load(os.getcwd() + "Total(Crop).npy"))  # 데이터들의 특징이 저장된 npy파일(드라이브에서 다운) , 보완가능

    def extract(self, img):
        img = img.resize((224, 224))
        img = img.convert('RGB')

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)


img = Image.open('test55_crop.jpg')  # 궁금한 이미지

fe = FeatureExtractor()
FEATURES = fe.features  # FEATRUES는 특징들이 들어가있는 Total.npy

query = fe.extract(img)

dists = np.linalg.norm(FEATURES - query, axis=1)
ids = np.argsort(dists)[:50]
print(ids)  # id 값은 csv에서 제일 처음 칼럼 값임. // 혹시라도 틀리면 +-1 수정을 해보기

# 이 밑에부분은 사용자가 확인을 위한 부분. img_paths는 그저 확인용도를 위한 것
# 서버에는 ids 만 넘겨줄 계획.
"""
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
