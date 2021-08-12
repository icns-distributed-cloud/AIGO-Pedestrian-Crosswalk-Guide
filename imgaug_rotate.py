import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import os

# Convert ( center of x, center of y, width, height ) to ( top left x, top left y, bottom right x, bottom right y )
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

## Convert ( top left x, top left y, bottom right x , bottom right y ) to ( center of x, center of y, width, height )
def convert_to_yolo_style(w,h,xmin,xmax,ymin,ymax):
    dw = 1./w
    dh = 1./h

    x = (xmax+xmin)/2.0
    y = (ymax+ymin)/2.0
    w = xmax-xmin
    h = ymax-ymin

    x = x*dw
    w = w*dw
    h = h*dh
    y = y*dh
    return x,y,w,h


exc = list()
def augmentation_rotate(img_path, label_path, save_path, keyword, rotate_scale):
    # Load

    img = cv2.imread(img_path)
    gt = np.loadtxt(label_path, dtype=np.float, delimiter=' ')
    if(len(gt)!=5):
        exc.append(img_path)
        return
    input_img = img[np.newaxis, :, :, :]
    img_w = img.shape[1]
    img_h = img.shape[0]
    label = gt[0]
    center_x = gt[1]
    center_y = gt[2]
    w = gt[3]
    h = gt[4]

    # Convert
    c_x1, c_y1, c_x2,c_y2 = convert_yolo_style_to_basic_bounding_box(img_w, img_h, center_x, center_y, w, h)
    bbox = [ia.BoundingBox(x1 = c_x1, y1 = c_y1, x2 = c_x2, y2 = c_y2)]

    # Roate
    seq = iaa.Sequential([
        iaa.Multiply((1, 1)),
        iaa.Affine(
            scale=(0.9),
            rotate=rotate_scale)])
    aug_img, aug_bbox = seq(images = input_img, bounding_boxes = bbox)
    draw = aug_bbox[0].draw_on_image(aug_img[0], size = 2, color = [0,0,255])
    res = np.hstack((img, draw))

    # Save the image
    for i, image_aug in enumerate(aug_img):
        cv2.imwrite(save_path+keyword+".jpg", image_aug)

    # Save the label
    x,y,w,h = convert_to_yolo_style(img_w,img_h,aug_bbox[0][1][0],aug_bbox[0][0][0],aug_bbox[0][1][1],aug_bbox[0][0][1])
    f = open(save_path+keyword+'.txt','w')
    f.write(str(int(label))+' '+str(x)+' '+str(y)+' '+str(w)+' '+str(h))
    f.close()


### SET
image_dir = '../origin_light/near_images/'
label_dir = '../origin_light/near_labels/'
output_dir = '../origin_light/test/'
rotate_scale = -30
keyword = 'tl_near_rotate'
keyword = keyword + str(rotate_scale) + '_'
# Load Images
images = os.listdir(image_dir)
cnt = 0
for image in images:
    origin_name = str(image).replace('.JPG', '')
    origin_name = str(origin_name).replace('.jpg', '')
    image_path = image_dir+image
    label_path = label_dir+origin_name+".txt"
    print(origin_name)
    augmentation_rotate(image_path,label_path,output_dir,keyword+str(cnt),rotate_scale)
    cnt += 1

# Read exception
f_exc = open('../origin_light/near_exc.txt','w')
for e in exc:
    f_exc.write(str(e)+'\n')
f_exc.close()