import os
import cv2 as cv
from PIL import Image
import resource
import sys
sys.setrecursionlimit(1500000000)
resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
labels_train = os.listdir('./Training')
labels_train = sorted(labels_train)
print(labels_train)
images_train = []
temp=[]
for filename in labels_train:
    temp = os.listdir('./Training/{}'.format(filename))
    temp_img=[]
    for img in temp:
        img=str(img)
        allowed_extension = 'ppm'
        if (img.split('.'))[1]==allowed_extension:
            path = './Training/{}/{}'.format(filename, img)
            im = cv2.imread(path)
            temp_img.append(im)
    images_train.append(temp_img)
