# Models Directory

This directory contains the trained models and encodings for the CCMS system.

## Required Files

### 1. headsetmodel.pt
- **Type**: YOLOv11 trained model
- **Purpose**: Detects headset presence near worker's face
- **Training**: Custom dataset with headset annotations
- **Performance**: Check training graphs in presentation

### 2. pdfinal.pt
- **Type**: YOLOv11 nano model
- **Purpose**: Detects phone usage near worker's face
- **Training**: 287 manually annotated images, augmented to 700+ using Roboflow
- **Training Platform**: Google Colab

### 3. face_encodings.pickle
- **Type**: Pickle file containing face encodings
- **Purpose**: Face recognition and identification
- **Generation**: Use `scripts/face_encodings_generator.py`
- **Structure**:
  ```python
  {
      "encodings": [encoding1, encoding2, ...],
      "names": ["person1", "person2", ...]
  }
  ```

## Model Versions

- YOLOv11 nano for phone detection
- YOLOv11 for headset detection
- face_recognition library for face encodings

## Performance Metrics

Refer to the training graphs and results in the project presentation (docs/CCMS_final_presentation.pdf)

## Updating Models

To retrain models:
1. Prepare annotated dataset
2. Use notebooks in `notebooks/` directory
3. Replace model files in this directory
4. Update configuration if needed