import os
from PIL import Image
import cv2
import numpy as np

dir_train = os.listdir('./Training')
dir_train = sorted(dir_train)
images_train = []
temp=[]
images_test =[]
temp_img_train = []
temp_img_test = []
temp_label_train = []
temp_label_test = []
for filename in dir_train:
    temp = os.listdir('./Training/{}'.format(filename))
    for img in temp:
        img=str(img)
        allowed_extension = 'ppm'
        if (img.split('.'))[1]==allowed_extension:
            path = './Training/{}/{}'.format(filename, img)
            im = Image.open(path)
            temp_label_train.append(filename)
            im = im.resize((28,28), Image.ANTIALIAS)
            im = np.array(im)
            im = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
            im.reshape(28,28,1)
            temp_img_train.append(im)
images_train = np.asarray(temp_img_train)
labels_train = np.asarray(temp_label_train)
dir_test = os.listdir('./Testing')
dir_test = sorted(dir_test)
for filename in dir_test:
    temp = os.listdir('./Testing/{}'.format(filename))
    for img in temp:
        img=str(img)
        allowed_extension = 'ppm'
        if (img.split('.'))[1]==allowed_extension:
            path = './Testing/{}/{}'.format(filename, img)
            temp_label_test.append(filename)
            im = Image.open(path)
            im = im.resize((28,28), Image.ANTIALIAS)
            im = np.array(im)
            im = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
            temp_img_test.append(im)
images_test = np.asarray(temp_img_test)
labels_test = np.asarray(temp_label_test)

