import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sklearn
from sklearn.metrics import confusion_matrix
import math

pdata0 = np.load("C:/Users/jonso/OneDrive/Desktop/Trinh Project/mm-fit/w00/w00_pose_2d.npy")
print(f'original pdata shape: {pdata0.shape}')
#trim last 4 joint data (right/left eye, right/left ear respectively)
pdata0 = pdata0[:,:,:-4]
#trim frame number
pdata0 = pdata0[:,:,1:]
print(f'new shape after trimming: {pdata0.shape}')
pdata0 = np.transpose(pdata0, (1, 2, 0))
print(f'new shape after transposition: {pdata0.shape}')
#print(pdata0[0])
#print(pdata0[3])

skel = np.load("C:/Users/jonso/OneDrive/Desktop/Trinh Project/Originals/skel00a.npy")
print(skel.shape)
print (np.array_equal(skel, pdata0))

n = "64"
s00 = np.load("file path for skeleton 00")
s01 = np.load("file path for skeleton 01")
s02 = np.load("file path for skeleton 02")
s03 = np.load("file path for skeleton 03")
s04 = np.load("file path for skeleton 04")
s05 = np.load("file path for skeleton 05")
s06 = np.load("file path for skeleton 06")
s06 = np.load("file path for skeleton 07")
s06 = np.load("file path for skeleton 08")
s06 = np.load("file path for skeleton 09")
s06 = np.load("file path for skeleton 10")
s06 = np.load("file path for skeleton 11")
s06 = np.load("file path for skeleton 12")
s06 = np.load("file path for skeleton 13")
s06 = np.load("file path for skeleton 14")
s06 = np.load("file path for skeleton 15")
s06 = np.load("file path for skeleton 16")
s06 = np.load("file path for skeleton 17")
s06 = np.load("file path for skeleton 18")
s06 = np.load("file path for skeleton 19")
s06 = np.load("file path for skeleton 20")

l00 = np.load("file path for label 00")
l00 = np.load("file path for label 01")
l00 = np.load("file path for label 02")
l00 = np.load("file path for label 03")
l00 = np.load("file path for label 04")
l00 = np.load("file path for label 05")
l00 = np.load("file path for label 06")
l00 = np.load("file path for label 07")
l00 = np.load("file path for label 08")
l00 = np.load("file path for label 09")
l00 = np.load("file path for label 10")
l00 = np.load("file path for label 11")
l00 = np.load("file path for label 12")
l00 = np.load("file path for label 13")
l00 = np.load("file path for label 14")
l00 = np.load("file path for label 15")
l00 = np.load("file path for label 16")
l00 = np.load("file path for label 17")
l00 = np.load("file path for label 18")
l00 = np.load("file path for label 19")
l00 = np.load("file path for label 20")

""" 
Testing data split:
train: ['01', '02', '03', '04', '06', '07', '08', '16', '17', '18'],
validation: ['14', '15', '19'],
testing: ['09', '10', '11'],
unseen testing: ['00', '05', '12', '13', '20']
"""

x_train = np.concatenate((s01, s02, s03, s04, s06, s07, s08, s16, s17, s18))
y_train = np.concatenate((l01, l02, l03, l04, l06, l07, l08, l16, l17, l18))
np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)
x_val = np.concatenate((s14, s15, s19))
y_val = np.concatenate((l14, l15, l19))
x_utest = np.concatenate((s00, s05, s12, s13, s20))
y_utest = np.concatenate((l00, l05, l12, l13, l20))

print(x_train.shape)
print(y_train.shape)

num_classes = 11
input_shape = (int(n), 14, 2)

x_train = np.concatenate((s01, s02, s03, s04, s06, s07, s08, s16, s17, s18))
y_train = np.concatenate((l01, l02, l03, l04, l06, l07, l08, l16, l17, l18))
x_val = np.concatenate((s14, s15, s19))
y_val = np.concatenate((l14, l15, l19))
x_train = np.vstack((x_train, x_val))
y_train = np.vstack((y_train, y_val))
x_test = np.concatenate((s09, s10, s11))
y_test = np.concatenate((l09, l10, l11))
x_cal = x_train
#scaling
x_train = x_train.astype("float32") / 2
x_test = x_test.astype("float32") / 2

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(x_test.shape)

y_train = keras.utils.to_categorical(y_train, num_classes)
y2_test = keras.utils.to_categorical(y_test, num_classes)
