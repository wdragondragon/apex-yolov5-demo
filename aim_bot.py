import os
import sys
import threading
import time
import tkinter
from ctypes import windll
from pathlib import Path

import pynput
import torch

from models.common import DetectMultiBackend
from tools import mouse_tools
from tools.mouse_tools import Win32ApiMouseMover
from utils.dataloaders import LoadScreenshots
from utils.general import check_img_size, non_max_suppression, scale_boxes, xyxy2xywh
from utils.torch_utils import select_device

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

mouse = pynput.mouse.Controller()


def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        shot_size=(320, 320)
):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    mouse_mover = Win32ApiMouseMover()
    threading.Thread(target=mouse_mover.start).start()
    resolution_x, resolution_y = get_resolution()
    current_mouse_x, current_mouse_y = mouse.position
    shot_width, shot_height = shot_size
    left_top_x, left_top_y = (resolution_x // 2 - shot_width // 2, resolution_y // 2 - shot_height // 2)  # 截图框的左上角坐标
    source = f"screen 0 {left_top_x} {left_top_y} {shot_width} {shot_height}"
    dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    for path, im, im0s, vid_cap, s in dataset:
        aims = []
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        pred = model(im, augment=augment, visualize=visualize)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        for i, det in enumerate(pred):
            im0 = im0s.copy()
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    line = (cls, *xywh)
                    if all(item != 'nan' for item in line):
                        aims.append(line)
        aim = get_nearest_center_aim(aims, current_mouse_x, current_mouse_y, shot_width, shot_height)
        if aim is None:
            continue
        move_x, move_y = calculate_mouse_offset(aim, resolution_x, resolution_y, left_top_x, left_top_y, shot_width,
                                                shot_height)
        mouse_mover.set_intention(move_x, move_y)


def get_nearest_center_aim(aims, current_mouse_x, current_mouse_y, shot_width, shot_height):
    """筛选离鼠标最近的label"""
    dist_list = []
    aims_copy = aims.copy()
    aims_copy = [x for x in aims_copy if x[0] == 0]
    if len(aims_copy) == 0:
        return
    for det in aims_copy:
        _, x_c, y_c, _, _ = det
        dist = (shot_width * float(x_c) - current_mouse_x) ** 2 + (shot_height * float(y_c) - current_mouse_y) ** 2
        dist_list.append(dist)
    return aims_copy[dist_list.index(min(dist_list))]


def get_resolution():
    """获取屏幕分辨率"""
    screen = tkinter.Tk()
    resolution_x = screen.winfo_screenwidth()
    resolution_y = screen.winfo_screenheight()
    screen.destroy()
    return resolution_x, resolution_y


def calculate_mouse_offset(aim, resolution_x, resolution_y, left_top_x, left_top_y, shot_width, shot_height):
    """计算鼠标偏移"""
    tag, target_x, target_y, target_width, target_height = aim
    target_shot_x = shot_width * float(target_x)  # 目标在截图范围内的坐标
    target_shot_y = shot_height * float(target_y)
    screen_center_x = resolution_x // 2
    screen_center_y = resolution_y // 2
    target_real_x = left_top_x + target_shot_x  # 目标在屏幕的坐标
    target_real_y = left_top_y + target_shot_y
    return int(target_real_x - screen_center_x), int(target_real_y - screen_center_y)


if __name__ == '__main__':
    run(weights=Path('apex_model/apex.engine'), data=Path('models/apex.yaml'), shot_size=(320, 320))
