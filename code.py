import os
from PIL import Image
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.initializers import glorot_normal
from keras.optimizers import Adam
from keras.models import model_from_json

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
            im = im.convert('RGB')
            temp_label_train.append(filename)
            im = im.resize((28,28), Image.ANTIALIAS)
            im = np.array(im)
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
            im = im.convert('RGB')
            im = im.resize((28,28), Image.ANTIALIAS)
            im = np.array(im)
            temp_img_test.append(im)
images_test = np.asarray(temp_img_test)
labels_test = np.asarray(temp_label_test)
images_test = images_test/255
images_train = images_train/255
print(len(images_test))
print(len(images_train))
#images_train = images_train.reshape(images_train.shape[0], 28, 28,1)
#images_test = images_test.reshape(images_test.shape[0], 28, 28, 1)
images_test = images_test.astype('float32')
images_train = images_train.astype('float32')
print('Training Samples:- {}'.format(images_train.shape[0]))
print('Testing Samples:- {}'.format(images_test.shape[0]))
print('Input shape = {} * {}'.format(images_train.shape[1], images_train.shape[2]))
int_dir_train = [x for x in range(len(dir_train))]
print(int_dir_train)
int_dir_test = [x for x in range(len(dir_test))]
print(int_dir_test)
int_dir_train = np_utils.to_categorical(int_dir_train, len(dir_train))
int_dir_test = np_utils.to_categorical(int_dir_test, len(dir_test))
print(len(int_dir_train))
for i in range(2):
    print(int_dir_train[i])
print(len(int_dir_test))

labels_train_int = []
for i in range(len(labels_train)):
    for j in range(len(dir_train)):
        if labels_train[i]==dir_train[j]:
            labels_train_int.append(int_dir_train[j])

labels_test_int = []
for i in range(len(labels_test)):
    for j in range(len(dir_test)):
        if labels_test[i]==dir_test[j]:
            labels_test_int.append(int_dir_test[j])
print(len(labels_train_int))
print(len(images_train))
model = Sequential()
model.add(Convolution2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer=glorot_normal(seed=None) , use_bias = True, bias_initializer='zeros', input_shape=(28, 28, 3)))
model.add(Convolution2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Convolution2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Convolution2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Convolution2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Convolution2D(128, kernel_size=(3, 3), activation='relu'))

model.add(Convolution2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(63))
model.add(Dropout(0.2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(np.array(images_train), np.array(labels_train_int), batch_size=64, nb_epoch=20, verbose=1, validation_data=(np.array(images_test), np.array(labels_test_int)))

score = model.evaluate(np.array(images_test), np.array(labels_test_int), verbose=0)
print('Test score: ', score[0])
print('Test accuracy: ', score[1])

im = Image.open('./index.jpeg')
im = im.convert('RGB')
im = im.resize((28,28), Image.ANTIALIAS)
im = np.array(im)
x = np.expand_dims(im, axis=0)

out1 = model.predict(x)
y = np.argmax(out1)
for i in range(len(dir_test)):
    if i==y:
        print(dir_train[i])
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
