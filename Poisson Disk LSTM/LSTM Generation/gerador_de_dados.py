import math
import random
from libpoisson import poisson_disc_samples
import cv2
import numpy as np
import imutils

def create_train(sample,l,id):
    for i in range(21,len(sample)):
        if i<(len(sample)-1):
            img = np.zeros([7000, 7000, 3], dtype=np.uint8)
            for j in range(0,i):
                cv2.circle(img, (int(sample[j][0]*1000), int(sample[j][1]*1000)), 500, (0, 0, 255), -1)
            img = imutils.resize(img,150)
            name_file = str(id)+"-"+str(i)+".jpg"
            cv2.imwrite(name_file, img)


r =1
n_ex = 10000
l = 7
samples = []

for i in range(0,n_ex):
    samples.append(poisson_disc_samples(l, l, r=r))
    if((len(samples[i])>20)):
        create_train(samples[i],l,i)
