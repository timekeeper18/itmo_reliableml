import io
import torch
import PIL
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F
import pandas as pd


class YoloV5Model():
    def __init__(self, pretrained_path: str = None):
        if pretrained_path:
            self.load_model(pretrained_path)
        else:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            # self.model.eval()

    def get_boxes(self, image) -> PIL.Image:
        image = Image.open(io.BytesIO(image))
        # tensor = F.to_tensor(image)
        results = self.model(image)
        boxes = results.pandas().xyxy[0]
        annotated_image = self.plot_boxes_on_image(image, boxes)

        return annotated_image

    def load_model(self, path: str):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=path, source='local')
        self.model.eval()

    def plot_boxes_on_image(self, image: PIL.Image, boxes: pd.DataFrame) -> PIL.Image:
        # Helper function to draw bounding boxes on the image using PIL.ImageDraw
        draw = ImageDraw.Draw(image)
        for _, box in boxes.iterrows():
            xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=2)

        return image

