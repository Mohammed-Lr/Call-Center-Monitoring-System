# Model Training Guide

## Phone Detection Model

### Dataset Preparation

1. **Manual Annotation**: 287 images of phone usage
2. **Augmentation**: Using Roboflow, augmented to 700+ images
3. **Annotations**: Bounding boxes around phones near faces

### Training Process

```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')

results = model.train(
    data='phone_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0
)
```

### Dataset YAML

```yaml
path: /path/to/dataset
train: images/train
val: images/val

nc: 1
names: ['phone']
```

### Training Platform

- **Platform**: Google Colab
- **GPU**: Tesla T4
- **Training Time**: ~2-3 hours
- **Model**: YOLOv11 nano

## Headset Detection Model

### Dataset Preparation

Custom dataset with headset annotations

### Training Process

```python
from ultralytics import YOLO

model = YOLO('yolo11.pt')

results = model.train(
    data='headset_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0
)
```

## Face Encodings

### Creating Face Dataset

1. Collect 3-5 clear images per person
2. Organize in directory structure:
   ```
   pickle_dataset/
   ├── person1/
   ├── person2/
   └── person3/
   ```

### Generate Encodings

```bash
python scripts/face_encodings_generator.py
```

## Using Training Notebooks

Located in `notebooks/`:
- `model_training_phone_detection.ipynb`
- `model_training_headset_detection.ipynb`

### Google Colab Setup

1. Upload notebook to Colab
2. Connect to GPU runtime
3. Upload dataset or mount Google Drive
4. Run training cells
5. Download trained model

## Evaluation Metrics

- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual positives
- **mAP**: Mean Average Precision
- **Loss**: Training and validation loss

Check graphs in the presentation PDF for detailed metrics.

## Fine-tuning

To improve model performance:

1. Increase dataset size
2. Adjust augmentation parameters
3. Tune hyperparameters
4. Use larger model variants (yolo11s, yolo11m)
5. Increase training epochs

## Export Models

```python
model = YOLO('best.pt')
model.export(format='onnx')
```