import os, sys, shutil

test_img = []
folder_from = r'C:\Users\sondors\Desktop\passports_all_02.12'
names = r'C:\Users\sondors\Desktop\400pass\recogn'
folder_to = r'C:\Users\sondors\Desktop\400pass\recognized'

for name in next(os.walk(names))[2]:
    
    test_img.append(name[:-4])

#print(test_img)
for i in range(len(test_img)):
    print(test_img[i])
    
    shutil.copyfile(os.path.join(folder_from, '{}'.format(test_img[i])), os.path.join(folder_to, '{}'.format(test_img[i])))

    os.remove(os.path.join(folder_from, '{}'.format(test_img[i])))