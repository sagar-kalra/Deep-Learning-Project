import os
from PIL import Image

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
            im = Image.open(path)
            temp_img.append(im)
    images_train.append(temp_img)
labels_test = os.listdir('./Testing')
labels_test = sorted(labels_test)
print(labels_test)
images_test = []
temp=[]
for filename in labels_test:
    temp = os.listdir('./Testing/{}'.format(filename))
    temp_img=[]
    for img in temp:
        img=str(img)
        allowed_extension = 'ppm'
        if (img.split('.'))[1]==allowed_extension:
            path = './Testing/{}/{}'.format(filename, img)
            im = Image.open(path)
            temp_img.append(im)
    images_test.append(temp_img)
