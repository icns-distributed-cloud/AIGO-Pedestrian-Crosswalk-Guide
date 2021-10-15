import sys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())
from collections import deque
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    apply_classifier, scale_coords, set_logging
from utils.torch_utils import select_device, load_classifier, time_sync


def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


@torch.no_grad()
def aigo_detect_test(weights='aigo-tl.pt',  # model.pt path(s)
        source='images/test.jpg',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.5,  # confidence threshold
        iou_thres=0.5,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        ):
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    res = 0
    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, pt, onnx = False, w.endswith('.pt'), w.endswith('.onnx')  # inference type
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        bs = 1  # batch_size

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        if pt:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
        elif onnx:
            img = img.astype('float32')
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        t1 = time_sync()
        if pt:
            pred = model(img, augment=augment, visualize=visualize)[0]
        elif onnx:
            pred = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_sync()

        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            s += '%gx%g ' % img.shape[2:]  # print string
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        res = 0 if c == 1 else 1
                        # set color
                        if (c == 1):
                            color = (0, 0, 255)
                        else:
                            color = (23, 154, 0)
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]}')
                        plot_one_box(xyxy, im0, label=label, color=color, line_thickness=line_thickness)
                        print("Result: "+names[c])
            else:
                return -1
            # Print time (inference + NMS)
            #print(f'{s}Done1. ({t2 - t1:.3f}s)')

            # Stream results
            if(view_img):
                cv2.imshow('test', im0)
                cv2.waitKey(1)  # 1 millisecond

    print(f'Finish: {time.time() - t0:.3f}s')
    return res

def aigo_tl_run():
    LIM = 10
    buff = deque()
    first_flag = 1
    curr_state = 1
    while(True):
        #print(curr_state)
        #print(buff)
        if(first_flag):
            try:
                res = aigo_detect_test(source='images/test.jpg')
            except Exception as e:
                print(e)
            buff.append(res)
            if(len(buff)>LIM):
                state = dict()
                first_flag=0
                state[1] = buff.count(1) # green
                state[0] = buff.count(0) # red
                state[-1] = buff.count(-1) # none
                curr_state = max(state, key=state.get)
        else:
            if(curr_state==-1): # none
                return -1
            elif(curr_state==1): # green ( -> red)
                try:
                    res = aigo_detect_test(source='images/test.jpg')
                except Exception as e:
                    print(e)
                if(len(buff)<20):
                    buff.append(res)
                else:
                    buff.popleft()
                    buff.append(res)
                if(buff.count(1)<buff.count(0)):
                    curr_state=0
            elif(curr_state==0): # red ( -> green -> go)
                try:
                    res = aigo_detect_test(source='images/test.jpg')
                except Exception as e:
                    print(e)
                if (len(buff) < 20):
                    buff.append(res)
                else:
                    buff.popleft()
                    buff.append(res)
                if (buff.count(1) > buff.count(0)):
                    return 1

