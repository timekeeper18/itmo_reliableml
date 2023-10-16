import torch
import PIL
from PIL import Image
from torchvision.transforms import functional as F


class YoloV5Model():

    def __init__(self, pretrained_path: str):
        if pretrained_path:
            self.load_model(pretrained_path)
        else:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model = self.model.autoshape()  # Set automatic shape compatibility
            self.model.eval()

    def get_boxes(self, image) -> PIL.Image:
        image = Image.open(image)
        tensor = F.to_tensor(image)
        results = self.model(tensor.unsqueeze(0))
        results.render()
        annotated_image = results.imgs[0]
        return annotated_image

    def load_model(self, path: str):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=path, source='local')
        self.model.eval()
