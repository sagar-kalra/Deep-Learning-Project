import os
from PIL import Image
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json
import os
import sys
sys.path.append('/home/')
import camera

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")
model.summary()
temp_label_test = []
temp_img_test =[]
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
            im = im.convert('RGB')
            im = im.resize((28,28), Image.ANTIALIAS)
            im = np.array(im)
            temp_img_test.append(im)
images_test = np.asarray(temp_img_test)
labels_test = np.asarray(temp_label_test)
images_test = images_test/255
labels_test_int = []
int_dir_test = [x for x in range(len(dir_test))]
int_dir_test = np_utils.to_categorical(int_dir_test, len(dir_test))
for i in range(len(labels_test)):
    for j in range(len(dir_test)):
        if labels_test[i]==dir_test[j]:
            labels_test_int.append(int_dir_test[j])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = model.evaluate(np.array(images_test), np.array(labels_test_int), verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

print('Do you want to capture your own image(1) or want to test it on a custom image(2). Choose 1 or 2.')
entry=int(input())
if entry==2:
    im = Image.open('./sample.ppm')
    im = im.convert('RGB')
    im = im.resize((28,28), Image.ANTIALIAS)
    im = np.array(im)
    x = np.expand_dims(im, axis=0)
    print(x.shape)
    x=x.reshape(1,28,28,3)

    out1 = model.predict(x)
    y = np.argmax(out1)
    dir_test = os.listdir('./Testing')
    dir_test = sorted(dir_test)
    for i in range(len(dir_test)):
        if i==y:
            print(dir_test[i])
elif entry == 1:
    camera.capture()
    im = Image.open('./captured_sign.png')
    im = im.convert('RGB')
    im = im.resize((28,28), Image.ANTIALIAS)
    im = np.array(im)
    x = np.expand_dims(im, axis=0)
    print(x.shape)
    x=x.reshape(1,28,28,3)

    out1 = model.predict(x)
    y = np.argmax(out1)
    dir_test = os.listdir('./Testing')
    dir_test = sorted(dir_test)
    for i in range(len(dir_test)):
        if i==y:
            print(dir_test[i])
    os.remove('./captured_sign.png')
else:
    print('Please choose the correct option and reexecute the code.')
