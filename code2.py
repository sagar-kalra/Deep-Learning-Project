import os
from PIL import Image
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

#score = model.evaluate(np.array(images_test), np.array(labels_test_int), verbose=0)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])

im = Image.open('./index2.png')
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
