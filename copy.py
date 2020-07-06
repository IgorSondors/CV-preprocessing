#!/bin/user/env python
#coding: utf-8
 
import os, sys, shutil

test_img = []
folder_from = '/home/sonders/Recognizer/server/not_api/images'
names = '/home/sonders/Recognizer/server/not_api/img_rec_txt'
folder_to = '/home/sonders/Recognizer/server/not_api/recognized'

for name in next(os.walk(names))[2]:
    
    test_img.append(name[:-4])



#print(test_img)
for i in range(len(test_img)):
    print(test_img[i])
    
    shutil.copyfile(os.path.join(folder_from, '{}'.format(test_img[i])), os.path.join(folder_to, '{}'.format(test_img[i])))

    os.remove(os.path.join(folder_from, '{}'.format(test_img[i])))

