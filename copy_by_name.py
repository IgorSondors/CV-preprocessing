'''from itertools import groupby

new_x = [el for el, _ in groupby(x)]
print(new_x)'''


test_img = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '11.jpg', '12.jpg', '13.jpg', '15.jpg', 'vertical-1.jpg.jpg', 'vertical-2.jpg.jpg', 'vertical-3.jpg.jpg', 'vertical-4.jpg.jpg', 'vertical-5.jpg.jpg', 'vertical-6.jpg.jpg', 'vertical-7.jpg.jpg', 'vertical-8.jpg.jpg', 'vertical-9.jpeg.jpg', 'vertical-10.jpeg.jpg', 'vertical-11.jpg.jpg', 'vertical-12.jpg.jpg', 'vertical-13.jpg.jpg', 'vertical-14.jpeg.jpg', 'vertical-15.jpg.jpg', 'DSC04651.JPG', 'DSC04653.JPG', 'DSC04655.JPG', 'DSC04656.JPG', 'DSC04657.JPG', 'DSC04658.JPG', 'DSC04660.JPG', 'DSC04661.JPG', 'DSC04662.JPG', 'DSC04663.JPG', 'IMG_20191128_155202.jpg', 'IMG_20191128_155302.jpg', 'IMG_20191128_155426.jpg', 'IMG_20191128_155525.jpg', 'IMG_20191128_160551.jpg', 'IMG_20191128_160715.jpg', 'IMG_20191128_160751.jpg', 'IMG_20191128_160859.jpg', 'IMG_20191128_160935.jpg', 'IMG_20191128_161019.jpg', 'IMG_20191128_161932.jpg', 'IMG_20191128_162009.jpg', 'IMG_20191128_162043.jpg', 'IMG_20191128_162131.jpg', 'IMG_20191128_162159.jpg', 'IMG_20191128_162427.jpg', 'IMG_20191128_162518.jpg', 'IMG_20191128_162603.jpg', 'IMG_20191128_162727.jpg', 'IMG_20191128_162805.jpg', 'IMG_20191128_162831.jpg', 'IMG_20191128_162901.jpg', 'IMG_20191128_163556.jpg', 'IMG_20191128_163716.jpg', 'IMG_20191128_163758.jpg', 'IMG_20191128_163824.jpg', 'IMG_20191128_163907.jpg', 'IMG_20191128_163932.jpg']

#!/bin/user/env python
#coding: utf-8
 
import os, sys, shutil
 
def CopyFile(fname, todir):
    result = False
    if not (os.path.isfile(fname)):
        print("Error: %s not found in current dir." % fname)
        return False
    try:        
        shutil.copyfile(fname, os.path.join(todir, fname))
    except shutil.Error as error:
        return False
    return True             
    
def main(): 
    if (len(sys.argv) == 3):
        if (os.path.isfile(sys.argv[1])):       
            todir = sys.argv[2]
            if not (os.path.isdir(todir)):
                os.mkdir(todir)
            print("File name %s" % sys.argv[1])
            with open(sys.argv[1], "r") as fin:         
                    for nfile in  [line.strip() for line in fin.readlines()]:
                        if (CopyFile(nfile, todir)):
                            print ("File %s success copy to %s" % (nfile, todir))
                        else:
                            print ("Error copy %s to %s" % (nfile, todir))
            print ("End file" )           
        else:
            print("File %s not found.Abort" % sys.argv[1])
            exit(0)
    else:
        print ("copylistfiles [x] Spouk//GNU :D\nuse: python files.py <list_file> <to_directory_need_copy>\n")
        
    
if  __name__ ==  "__main__" :    
    main()

for i in range(len(test_img)):
    CopyFile(test_img[i], r'C:\Users\sondors\Desktop\images\test')