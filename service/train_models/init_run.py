from pathlib import Path

from ultralytics import YOLO, settings

# Settings init
DATASET = 'Occlusal caries detection.v10i.yolov8'
ROOT = Path().cwd()
DATA = ROOT / 'service' / 'data'
settings.update({'runs_dir': ROOT / 'service' / 'train_models' / 'runs',
                 'wandb': True})

# Create a new YOLO model from scratch
# model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO(ROOT / 'service' / 'train_models' / 'pretrained' / 'yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
model.train(data=DATA / DATASET / 'data.yaml',
            task='detect',
            epochs=3)

# Evaluate the model's performance on the validation set
results = model.val()
print("----------------------------- Val results ----------------------------")
print(results)
print("----------------------------- ----------- ----------------------------")

# Export the model to ONNX format
success = model.export(format='onnx')

# Perform object detection on an image using the model
results = model(DATA / DATASET / 'valid' / 'images' / '2019-Picture11-mand_png.rf.6c45cb321e852364ff672316e992a27c.jpg')
print("----------------------------- Prediction ----------------------------")
print(results)

# Reset settings to default values
settings.reset()
