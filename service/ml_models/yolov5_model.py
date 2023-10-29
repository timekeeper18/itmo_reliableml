import io
import torch
import PIL
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F
import onnx
import onnxruntime
from onnx2torch import convert
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from yolov5.utils.general import non_max_suppression, xyxy2xywh
from typing import Tuple


class YoloV5Model():
    def __init__(self, pretrained_path: str = None):
        self.onnx = False
        if pretrained_path:
            if Path(pretrained_path).suffix == '.onnx':
                self.onnx = True
                self.session = onnxruntime.InferenceSession(pretrained_path, providers=['CPUExecutionProvider'])
            else:
                self.load_model(pretrained_path)
        else:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            # self.model.eval()

    def get_boxes(self, image) -> PIL.Image:
        if self.onnx:
            img = np.fromstring(image, np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h_ratio = img.shape[0] / 640
            w_ratio = img.shape[1] / 640
            # resized = img.astype(np.float32)
            resized = cv2.resize(img, (640,640), interpolation = cv2.INTER_AREA).astype(np.float32)
            resized = resized.transpose((2, 0, 1))
            resized = np.expand_dims(resized, axis=0)  # Add batch dimension                
            ort_outputs = self.session.run([], {'input': resized})[0]

            output = torch.from_numpy(np.asarray(ort_outputs))
            out = non_max_suppression(output, conf_thres=0.25, iou_thres=0.45)[0]

            # convert xyxy to xywh
            xyxy = out[:,:4]
            boxes = xyxy[0]
            xywh = xyxy2xywh(xyxy)
            out[:, :4] = xywh     
            annotated_image = self.plot_boxes_on_image(Image.open(io.BytesIO(image)), out, (h_ratio, w_ratio))            
        else:
            image = Image.open(io.BytesIO(image))            
            # tensor = F.to_tensor(image)
            results = self.model(image)
            boxes = results.pandas().xyxy[0]
            annotated_image = self.plot_boxes_on_image(image, boxes)

        return annotated_image

    def load_model(self, path: str):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=path, source='local')
        self.model.eval()

    def plot_boxes_on_image(self, image: PIL.Image, boxes: pd.DataFrame, ratio: Tuple = None) -> PIL.Image:
        # Helper function to draw bounding boxes on the image using PIL.ImageDraw
        draw = ImageDraw.Draw(image)
        if self.onnx:
            # tmp_image = Image.fromarray(image)
            h_ratio, w_ratio = ratio
            scaled_boxes = self.scale_boxes((640, 640), boxes, image.size)
            for i, (x, y, w, h, score, class_id) in enumerate(scaled_boxes):
                # real_x = x * w_ratio # resize from model size to image size
                # real_y = y * h_ratio

                # shape = (real_x, real_y, (x + w) * w_ratio, (y + h) * h_ratio) # shape of the bounding box to draw               
                shape = (x, y, x + w, y + h) # shape of the bounding box to draw          
                color = 'red'
                draw.rectangle(shape, outline=color, width=4)
                fnt = ImageFont.load_default()
                draw.multiline_text((x + 8, y + 8), f"{score*100:.2f}%", font=fnt, fill=color)
        else:
            for _, box in boxes.iterrows():
                xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
                draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=2)

        return image
    
    def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None):
        # Rescale boxes (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
        boxes[..., :4] /= gain
        self.clip_boxes(boxes, img0_shape)
        return boxes

    def clip_boxes(self, boxes, shape):
        # Clip boxes (xyxy) to image shape (height, width)
        if isinstance(boxes, torch.Tensor):  # faster individually
            boxes[..., 0].clamp_(0, shape[1])  # x1
            boxes[..., 1].clamp_(0, shape[0])  # y1
            boxes[..., 2].clamp_(0, shape[1])  # x2
            boxes[..., 3].clamp_(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
