import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import sys
from PIL import Image
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


#f = sys.argv[1]


saved = load_model("save_ckp_frozen.h5")


class fashion_tools(object):
    def __init__(self, imageid, model, version=1.1):
        self.imageid = imageid
        self.model = model
        self.version = version

    def get_dress(self, stack=False):
        """limited to top wear and full body dresses (wild and studio working)"""
        """takes input rgb----> return PNG"""
        name = self.imageid
        file = cv2.imread(name)
        file = tf.image.resize_with_pad(
            file, target_height=512, target_width=512)
        rgb = file.numpy()  # 이미지파일을 numpy로 변환 후 작업.. 이작업을 생략 가능할까
        file = np.expand_dims(file, axis=0) / 255.
        seq = self.model.predict(file)
        seq = seq[3][0, :, :, 0]
        seq = np.expand_dims(seq, axis=-1)
        c1x = rgb*seq
        c2x = rgb*(1-seq)
        cfx = c1x+c2x
        dummy = np.ones((rgb.shape[0], rgb.shape[1], 1))
        rgbx = np.concatenate((rgb, dummy*255), axis=-1)
        rgbs = np.concatenate((cfx, seq*255.), axis=-1)
        if stack:
            stacked = np.hstack((rgbx, rgbs))
            return stacked
        else:
            return rgbs

    def get_patch(self):
        return None


# running code

for i in range(1, 32):  # 잠시 90개만함. 근데 이것도 시간이 좀 걸림. 각 장당 2~3초
    f = "./Top_extract_3/"+str(i)+".jpg"  # 사진 있는 곳
    api = fashion_tools(f, saved)
    image_ = api.get_dress(stack=True)
    cv2.imwrite("./Test"+str(i)+".png", image_)  # 임시 저장용사진
    image1 = Image.open("./Test"+str(i)+".png")
    # image1.show()

    croppedImage = image1.crop((512, 0, 1024, 512))
    croppedImage.save('./Test'+str(i)+'.PNG')  # 짤린 사진.
   # croppedImage.show()
#print("잘려진 사진 크기 :", croppedImage.size)


#image1 = Image.open('Result.png')
# image1.show()

# 이미지의 크기 출력
# print(image1.size)

# 이미지 자르기 crop함수 이용 ex. crop(left,up, rigth, down)
#croppedImage = image1.crop((512, 0, 1024, 512))

# croppedImage.show()

#print("잘려진 사진 크기 :", croppedImage.size)

# croppedImage.save('croppedImage.PNG')


#directory = sys.argv[2]

#new_img_file = img_file.split('/')[-1]
#cv2.imwrite(directory+"/"+str(new_img_file), image_)
