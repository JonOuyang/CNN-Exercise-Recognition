import numpy as np
import math
import statistics as st
from statistics import mode

#Function for batch size 128 preprocessing
def convert128(testSkel):
    global y, nSkel, nLabel
    y = int(skel.shape[0]/64)-2
    nSkel = np.empty([y, 128, 14, 2])
    nLabel = np.empty([y, 1])
    for i in range(0, y):
        nSkel[i] = skel[(i*64):(i*64)+128,:,:]
        nLabel[i] = st.mode(label.flatten()[(i*64):(i*64)+128])
    print(f'Skeleton data shape: {nSkel.shape}')
    print(f'Label data shape: {nLabel.shape}')

#Function for batch size 64 preprocessing
def convert64(testSkel):
    global y, nSkel, nLabel
    y = int(skel.shape[0]/32)-2
    nSkel = np.empty([y, 64, 14, 2])
    nLabel = np.empty([y, 1])
    for i in range(0, y):
        nSkel[i] = skel[(i*32):(i*32)+64,:,:]
        nLabel[i] = st.mode(label.flatten()[(i*32):(i*32)+64])
    print(f'Skeleton data shape: {nSkel.shape}')
    print(f'Label data shape: {nLabel.shape}')
    return y, nSkel, nLabel

#Function for batch size 32 preprocessing
def convert32(testSkel):
    global y, nSkel, nLabel
    y = int(skel.shape[0]/16)-2
    nSkel = np.empty([y, 32, 14, 2])
    nLabel = np.empty([y, 1])
    for i in range(0, y):
        nSkel[i] = skel[(i*16):(i*16)+32,:,:]
        nLabel[i] = st.mode(label.flatten()[(i*16):(i*16)+32])
    print(f'Skeleton data shape: {nSkel.shape}')
    print(f'Label data shape: {nLabel.shape}')
    return y, nSkel, nLabel
  
#All data file paths have been replaced with "file directory." 
#Replace with the real data file path
label = np.load("Label Data File Directory")
skel = np.load("Skeleton Data File Directory")
print(f'original skel shape: {skel.shape}')
print(f'original label shape: {label.shape}')
print(f'original shape: {skel.shape}')

#removes last 2D arrays to make divisible by 128
c=0
while skel.shape[0]%128 != 0:
    skel = np.delete(skel, -1, axis=0)
    label = np.delete(label, -1, axis = 0)
    c += 1
print(f'new shape (skel): {skel.shape}')
print(f'new shape (label): {label.shape}')
print(f'A total of {c} elements have been removed')
y=0
nSkel = 0
nLabel = 0
convert128(skel)

#Saves the preprocessed labels, with the number in parenthesis representing the batch size
np.save('labels20(128).npy', nLabel)
np.save('skel20a(128).npy', nSkel)
