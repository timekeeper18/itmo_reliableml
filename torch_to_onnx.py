import os
import time
import torch
from PIL import Image
import torch.onnx
import onnxruntime
import onnx
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from matplotlib.pyplot import imshow


# Путь к нашим исходным весам.
path_to_weights = "pretrained/best/best.pt"

# Загрузим веса в модель.
model = torch.hub.load('ultralytics/yolov5', 'custom', autoshape=False, path=path_to_weights)

image = Image.open("input.png", mode='r', formats=None)
image = image.resize((640, 640))

input_image = torch.unsqueeze(torch.tensor(np.array(image)), 0).float()
input_image = torch.reshape(input_image, (input_image.shape[0], input_image.shape[3], input_image.shape[1], input_image.shape[2]))

# конвертируем нашу модель в ONNX.
torch.onnx.export(model,               # model being run
                  input_image,                         # model input (or a tuple for multiple inputs)
                  "pretrained/best.onnx",   # where to save the model (can be a
                  # file
                  # or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=13,          # the ONNX version to export
                  # the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})