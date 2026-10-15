import numpy as np
import os
from PIL import Image

def image_read_cv2(path, mode='RGB'):
    image = Image.open(path)
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = np.array(image.convert('RGB')).astype('float32')
    elif mode == 'GRAY':
        img = np.array(image.convert('L')).astype('float32')
    elif mode == 'YCrCb':
        img = np.array(image.convert('YCbCr')).astype('float32')
    return img

def img_save(image,imagename,savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    array = np.clip(np.round(image), 0, 255).astype(np.uint8)
    Image.fromarray(array).save(os.path.join(savepath, "{}.png".format(imagename)))
