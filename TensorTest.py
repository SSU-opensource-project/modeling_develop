from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from pathlib import Path
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# 특징잡는 클래스


class FeatureExtractor:
    def __init__(self):
        # VGG16모델(CNN 흔한 모델) imagenet사용 (데이터셋)
        base_model = VGG16(weights='imagenet')

        # Customize the model to return features from fully-connected layer
        self.model = Model(inputs=base_model.input,  # FC layer
                           outputs=base_model.get_layer('fc1').output)
     #   self.features = np.array(np.load(os.getcwd() + "/mFeatures.npy"))

    def extract(self, img):
        # 이미지크기 통일
        img = img.resize((224, 224))

        # Convert the image color space
        img = img.convert('RGB')
        # Reformat the image
        x = image.img_to_array(img)  # 이미지를 배열로
        x = np.expand_dims(x, axis=0)  # x배열 차원을 늘림
        x = preprocess_input(x)  # x를 모델에 넣기 위한 형태로 전처리를 함

        # Extract Features
        feature = self.model.predict(x)[0]  # x는 배열, [0]은 첫번쨰 예측feature은
        return feature / np.linalg.norm(feature)  # feature은 벡터?


fe = FeatureExtractor()
features = []
img_paths = []

# 데이터베이스 이미지로 Image 특징 벡터를 저장
# Save Image Feature Vector with Database Images
for i in range(1, 270):  # 값 변경가능
    j = i//90
    if i % 100 == 0:

        print(i)
    try:
        image_path = "./Pictures/상의" + \
            str(j+1)+" Page " + str(i//(j+1)) + ".jpg"
        img_paths.append(image_path)

        # Extract Features
        feature = fe.extract(img=Image.open(image_path))

        features.append(feature)

        # Save the Numpy array (.npy) on designated path
        feature_path = "./Feature/" + str(i+1) + ".npy"
        np.save(feature_path, feature)  # Feature 폴더에 npy 파일 저장함.
    except Exception as e:
        print('예외가 발생했습니다.', e)


# 궁금한 데이터 입력
img = Image.open("testData.jpg")

# 쿼리에서 특징 추출
query = fe.extract(img)

# Calculate the similarity (distance) between images
dists = np.linalg.norm(features - query, axis=1)  # axis가 1인이유 : 모름.. ㅜㅜ

# 가장 가까운 30개의 이미지 추출
ids = np.argsort(dists)[:30]


scores = [(dists[id], img_paths[id], id) for id in ids]

# Visualize the result
axes = []
fig = plt.figure(figsize=(8, 8))
for a in range(5*6):
    score = scores[a]
    axes.append(fig.add_subplot(5, 6, a+1))
    subplot_title = str(round(score[0], 2)) + "/m" + str(score[2]+1)
    axes[-1].set_title(subplot_title)
    plt.axis('off')
    plt.imshow(Image.open(score[1]))
fig.tight_layout()
plt.show()
