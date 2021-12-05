import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import sys
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


f = sys.argv[1]


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
        rgb = file.numpy()
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


api = fashion_tools(f, saved)
image_ = api.get_dress(stack=True)
cv2.imwrite("Result.png", image_)

#directory = sys.argv[2]

#new_img_file = img_file.split('/')[-1]
#cv2.imwrite(directory+"/"+str(new_img_file), image_)
