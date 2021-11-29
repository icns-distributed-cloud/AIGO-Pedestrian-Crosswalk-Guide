import numpy as np
import cv2
import os

def convert_yolo_style_to_basic_bounding_box(img_w, img_h, center_x, center_y, w,h):
    dw = 1. / img_w
    dh = 1. / img_h

    center_x = center_x / dw
    center_y = center_y / dh

    w = round(w / dw)
    h = round(h / dh)

    x1 = round((2 * center_x - w) / 2)
    y1 = round((2 * center_y - h) / 2)

    x2 = round((2 * center_x + w) / 2)
    y2 = round((2 * center_y + h) / 2)

    return x1,y1,x2,y2
# color
RED = (0,0,255)

# path
image_path = '{path}'
label_path = '{path}'
background_img = '{path}'
save_path = '{path}'

# read background image
back_img = cv2.imread(background_img,cv2.IMREAD_UNCHANGED)

# read labels
files = os.listdir(image_path)
for file in files:
    # get the origin name
    origin_name = str(file).replace('.JPG', '')
    origin_name = str(origin_name).replace('.jpg', '')

    # read the label
    f = open(label_path+origin_name+'.txt')
    line = f.readline()
    line = line.split(' ')
    print(line)

    # get yolo style bound
    center_x = float(line[1])
    center_y = float(line[2])
    w = float(line[3])
    h = float(line[4])
    f.close()

    # get the image to check the size of image
    img = cv2.imread(image_path + file, cv2.IMREAD_UNCHANGED)
    img_w = img.shape[1]
    img_h = img.shape[0]
    print(img_w,img_h)

    # check the size
    if(img_w != 4032 ):
        continue

    # convert to bounding box
    x1, y1, x2, y2 = convert_yolo_style_to_basic_bounding_box(img_w, img_h, center_x, center_y, w, h)
    back_img = cv2.rectangle(back_img, (x1, y1), (x2, y2), RED, 3)


cv2.imshow('result',back_img)
cv2.imwrite(save_path,back_img)
cv2.waitKey(0)
cv2.destroyWindow()
