import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os

#path = '21.jpg'
dir = r'C:/Users/sondors/Desktop/ready'

def Rotation90(path):
    img = cv2.imread(path,3)
    print(np.shape(img))
    img = np.rot90(img)
    cv2.imwrite("vertical-" + str(i) + ".jpg", img)


for i in os.listdir(dir):
    path = os.path.join(dir,i)
    print(i, path)
    Rotation90(path)

'''def Rotation180(path):
    img = cv2.imread(path,3)
    print(np.shape(img))
    img = np.rot90(img)
    img = np.rot90(img)
    cv2.imwrite(path, img)

for i in os.listdir(dir):
    path = os.path.join(dir,i)
    print(i, path)
    Rotation180(path)'''