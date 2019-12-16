import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os

def intersection_lines_method2(line1, line2):
    '''
    Returns closest integer pixel locations.
    See Wiki: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    '''
    import numpy as np
    x1, y1, x2, y2 = line1[0]

    a1 = (y1 - y2)
    b1 = (x2 - x1) + 0.00001
    c1 = (x1*y2 - x2*y1)

    x3, y3, x4, y4 = line2[0]

    a2 = (y3 - y4)
    b2 = (x4 - x3) + 0.00001
    c2 = (x3*y4 - x4*y3)

    x0 = (- c2 / b2 + c1 / b1) / (- a1 / b1 + a2 / b2)
    y0 = - x0 * (a1 / b1) - c1 / b1

    if x0 >= 0 and y0 >= 0:
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [[x0, y0]]

    return None


def UseCannyFilter(image):
    h, w = np.shape(image)[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (5, 5))
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    dst = cv2.Canny(blur, 5, 100)

    #cv2.imshow('im2', image)
    #cv2.waitKey(0)
    #cv2.imshow('im', dst)
    #cv2.waitKey(0)

    lines = []
# Получаем линии с помощью преобразования Хафа
    lines2 = cv2.HoughLines(dst, 1, np.pi/180, 100)
    for line in lines2:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 5000*(-b))
            y1 = int(y0 + 5000*(a))
            x2 = int(x0 - 5000*(-b))
            y2 = int(y0 - 5000*(a))
            lines.append([[x1, y1, x2, y2]])
    if lines is None:
        return None
    #print('lines is:', lines)
# Разделяем линии на горизонтальные и вертикальные
    v_lines = [x for x in lines if np.abs(x[0][3] - x[0][1]) >= 2*np.abs(x[0][2] - x[0][0])]
    h_lines = [x for x in lines if np.abs(x[0][3] - x[0][1]) < 0.1*np.abs(x[0][2] - x[0][0])]
    #print('v is:', v_lines)
    #print('h is:', h_lines)

# Находим точки пересечения
    intersection_points = []
    for v_line in v_lines[:15]:
        cv2.line(image, (v_line[0][0], v_line[0][1]), 
                (v_line[0][2], v_line[0][3]), (255, 0, 0), 5)
        for h_line in h_lines[:15]:
            cv2.line(image, (h_line[0][0], h_line[0][1]), 
                (h_line[0][2], h_line[0][3]), (0, 0, 255), 5)
            point = intersection_lines_method2(v_line, h_line)
            if point is not None:
                intersection_points.append(point)
# Рисуем изображение с линиями Хафа
    #cv2.imshow('im2', image)
    #cv2.waitKey(0)

    if not intersection_points:
        return None
    
    #print('intersection_points is:', intersection_points)

    source_points_list = np.array([[x, y] for [[x, y]] in intersection_points])

    #print('source_points_list is:', source_points_list)

    # Вот этот метод
    result = []
    for i in range(4):
        if i==0:
            source_points_list = sorted(
                source_points_list, key=lambda x: np.sqrt(np.power((x[0] - 0), 2) + np.power((x[1] - 0), 2)))
            result.append(source_points_list[0])
        elif i==1:
            source_points_list = sorted(
                source_points_list, key=lambda x: np.sqrt(np.power((x[0] - 0), 2) + np.power((x[1] - h), 2)))
            result.append(source_points_list[0])
        elif i==2:
            source_points_list = sorted(
                source_points_list, key=lambda x: np.sqrt(np.power((x[0] - w), 2) + np.power((x[1] - h), 2)))
            result.append(source_points_list[0])
        elif i==3:
            source_points_list = sorted(
                source_points_list, key=lambda x: np.sqrt(np.power((x[0] - w), 2) + np.power((x[1] - 0), 2)))
            result.append(source_points_list[0])
        
    return np.array(result)

    #print('result is:', np.array(result))
def UseTreshold(image):
    h,w = np.shape(image)[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print(np.mean(gray))
    if np.mean(gray) < 120:
        gray = cv2.bitwise_not(gray)

    q, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    

'''# Загружаем изображение паспорта
img = cv2.imread('141.jpg',3)

# Вертикальный отступ
img = np.vstack((img, np.full((30, np.shape(img)[1]), 255, dtype=np.uint8) ))
img = np.vstack((np.full((30, np.shape(img)[1]), 255, dtype=np.uint8), img ))

# Горизонтальный отступ

img = np.hstack((img, np.full((np.shape(img)[0], 30), 255, dtype=np.uint8) ))
img = np.hstack((np.full((np.shape(img)[0], 30), 255, dtype=np.uint8), img ))



'''

def ShowWarp(path):
    image = cv2.imread(path,3)
    #print(path)
    res = UseCannyFilter(image)

    #print('res is:', res)
    res = list(res)
    res[1], res[3] = res[3], res[1]

    #print('res_ord is:', res)

    '''plt.imshow(image)
    plt.show()'''

    # Поиск длины и ширины варпа
    x = [res[i][0] for i in range(4)]
    x_max = max(x) - min(x)
    y = [res[i][1] for i in range(4)]
    y_max = max(y) - min(y)

    # Делаем варп преобразование исходного изображения
    pts1 = np.float32(res)
    pts2 = np.float32([[0,0],[x_max,0],[x_max,y_max],[0,y_max]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    img = cv2.imread(path,3)
    print(np.shape(img))
    print(np.shape(image))
    dst = cv2.warpPerspective(img,M,(x_max,y_max))
    if np.shape(image)[0]>np.shape(image)[1]:
        dst = np.rot90(dst)

    #cv2.imshow('warp', dst)
    cv2.imwrite(path, dst) # замена файла его варпнутой версией
    #cv2.waitKey(0)



dir = r'C:/Users/sondors/Desktop/Labelling'

for i in os.listdir(dir):
    path = os.path.join(dir,i)
    print(i, path)
    ShowWarp(path)


