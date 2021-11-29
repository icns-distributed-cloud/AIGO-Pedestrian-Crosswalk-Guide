#pip3 install opencv-python
import os
import cv2
import shutil
import numpy as np

### SET
input_dir = './input/'
label_dir = './label/'
output_dir = './output/'
keyword = 'KEYWORD'

# Add image augmentation such as rotation and brightness control
cnt = 0
val = 30 # val
#if output_dir not in os.listdir():
#    os.mkdir(output_dir)

images = os.listdir(input_dir)
print(len(os.listdir(output_dir)))
for image in images:
    print(image)
    img = cv2.imread(input_dir + "/" + image)
    array = np.full(img.shape, (val, val, val), dtype=np.uint8)

    origin_name = str(image).replace('.JPG', '')
    origin_name = str(image).replace('.png', '')
    origin_name = str(origin_name).replace('.jpg', '')

    # set
    add_dst = cv2.add(img, array)
    sub_dst = cv2.subtract(img, array)
    print(cnt)
    # show
    # cv2.imshow('img',img)
    # cv2.imshow('add',add_dst)
    # cv2.imshow('sub',sub_dst)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    cv2.imwrite(output_dir + "/" + keyword + "add" + str(cnt) + ".jpg", add_dst)
    cv2.imwrite(output_dir + "/" + keyword + "sub" + str(cnt) + ".jpg", sub_dst)
    shutil.copy(label_dir+origin_name+'.txt', output_dir + keyword + "add" + str(cnt) +'.txt')
    shutil.copy(label_dir + origin_name + '.txt',
                output_dir + keyword + "sub" + str(cnt) + '.txt')
    cnt = cnt + 1
