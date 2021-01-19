#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# util
import numpy as np
import time
from PIL import Image
from config import IMG_H, IMG_W


def get_seed():
    t      = str(time.time()).split('.')
    first  = t[0][-7:]
    second = t[1][:2]
    if len(second) < 2:
        second = second + '0' * (2 - len(second))
    return int(first + second)


def gray_scale(im):
    im = Image.fromarray(im).convert("L")
    im = np.array(im).astype("uint8")
    return im[:, :, np.newaxis]


def resize(im):
    im = Image.fromarray(im.astype(np.uint8))
    im = im.resize((IMG_W, IMG_H), Image.ANTIALIAS)
    im = np.array(im).astype("uint8")
    return im


def resize_gray(im):
    # combine two functions above
    return gray_scale(resize(im))


