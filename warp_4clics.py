import cv2
import numpy as np
import os

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy
    
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),30,(255,0,0),-1)
        ix,iy = [x,y]
        dots.append([x,y])
    '''
    if event == ord("r"):
        dots.pop()'''
    
ix,iy = -1,-1

#path = r'C:/Users/sondors/Desktop/cropped/1.jpg'

dir = r'C:\Users\sondors\Desktop\passp_add'

for i in os.listdir(dir):
    path = os.path.join(dir,i)
    print(path)


    img = cv2.imread(path, 3)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image',draw_circle)
    dots = []
    while(1):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image',img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
    print(dots)
    dots = dots[:4]
    print(dots)
    # Поиск длины и ширины варпа
    x = [dots[i][0] for i in range(4)]
    x_max = max(x) - min(x)
    y = [dots[i][1] for i in range(4)]
    y_max = max(y) - min(y)
    print(x_max, y_max)
    # Делаем варп преобразование исходного изображения
    pts1 = np.float32(dots)
    pts2 = np.float32([[0,0],[x_max,0],[x_max,y_max],[0,y_max]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    img = cv2.imread(path,3) 
    dst = cv2.warpPerspective(img,M,(x_max,y_max))
    #print('dst is:', dst)
    #cv2.imshow('warp', dst)

    cv2.imwrite(path, dst) # замена файла его варпнутой версией
    #cv2.waitKey()

