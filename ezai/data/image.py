# image codes
"""
Notes:

    # image formats: M x N x 3 x 1 [3 is color, 1 is alpha channel)
    plt.imshow : RGB
    cv2.imshow : BGR
    cameras: some output GBR

from PIL image Image
size = (64, 64)
my_img = Image.open(‘blahblah.jpg’)
my_img.show()
my_img.thumbnail(size)
my_img.save(output file, “JPEG")
2)OpenCV, the open source computer version library, it also does image processing. But have more like capture from cam, OpenCV is implemented using cv2 and Numpy.

→To load, display, save an image looks like below:

import cv2
my_img = cv2.imread(‘blahblah.jpg’, 0)
cv2.imshow(‘displaymywindows', my_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(’saveto.png’, img)

Conversion between PIL and OpenCV image

i)image: from PIL to OpenCV, something like below:

from PIL import Image
import cv2
import numpy
pil_image = PIL.Image.open(‘Image.jpg’).convert(‘RGB')
opencv_image =numpy.array(pil_image)
ii) image: from OpenCV to PIL, something like below:

import cv2, Image
captured = cv2.VideoCapture(0)
myimage = captured.read()
pil_image = Image.fromarray(myimage)
pil_image.show()

"""
import cv2
import numpy as np

import math
import itertools
import urllib.parse
import urllib.request

def img2mask(filename: str):
    char_mask = np.array(cv2.imread(filename,0))  # o reads in greyscale, 1 in color
    return char_mask


def load_image(path, cv2_flag = cv2.IMREAD_UNCHANGED):
    """ Load an image in bgr, returns numpy array """
    # if the path appears to be an URL
    if urllib.parse.urlparse(path).scheme in ('http', 'https'):
        # set up the byte stream
        img_stream = np.asarray(bytearray(urllib.request.urlopen(path).read()))
        img = cv2.imdecode(img_stream, cv2_flag)
    else:
        # else use it as local file path
        img = cv2.imread(path, cv2_flag)
    return img


"""
operations on memory-contiguous arrays are most efficient. 
In particular, OpenCV in-place operations require a contiguous array to 
avoid unexpected results. 
The safest approach is to always make a copy of the array as in the examples below.
"""
# also works for rgb2bgr
def bgr2rgb(img, copy=True):
    img = img[...,::-1]
    img = img.copy() if copy else img
    return img # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def rgb2gbr(img, copy=True):
    img = img[...,[2,0,1]]
    img = img.copy() if copy else img
    return img

