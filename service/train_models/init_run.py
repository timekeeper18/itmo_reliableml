from pathlib import Path

import wandb
from ultralytics import YOLO, settings

# Settings init
ROOT = Path().cwd()

DATASET = 'dentasis_v30_2_Occlus_Caries_v0'
DATA = ROOT / 'service' / 'data' / DATASET

wandb.login(key='a427b2ba60f438b85345fe30f56d9f8fd6802294')
run = wandb.init(
    entity="dentist_ai",
    project="Dentist_AI",
    name=f'{DATASET}_YOLO8',)


settings.update({'runs_dir': ROOT / 'service' / 'train_models' / 'runs',
                 'wandb': True})

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov5m6.yaml')
# model = YOLO(ROOT / 'service' / 'train_models' / 'pretrained' / 'yolov8n.pt')

if DATA.is_dir():
    artifact_dir = DATA
else:
    artifact = run.use_artifact("dentist_ai/Dentist_AI/dentasis_v30_2_Occlus_Caries_v0:v0", type="dataset")
    artifact.download(DATA)

# Train the model using the 'coco128.yaml' dataset for 3 epochs
model.train(data=DATA / 'data.yaml',
            optimizer='SGD',
            task='detect',
            batch=4,
            epochs=300)

# Evaluate the model's performance on the validation set dentasis_v30_4_Dental_disease_detection_Occlus_v0
results = model.val()
print("----------------------------- Val results ----------------------------")
print(results)
print("----------------------------- ----------- ----------------------------")

# Export the model to ONNX format
success = model.export(format='onnx')

# Perform object detection on an image using the model
# results = model(DATA / 'valid' / 'images' / '1_jpeg.rf.6e6037fef3b0f8023be0f9a9d715980b.jpg')
# print("----------------------------- Prediction ----------------------------")
# print(results)

# Reset settings to default values
# settings.reset()
