# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Webcam (for real-time monitoring)
- GPU (optional, for faster processing)

## Step 1: Clone Repository

```bash
git clone https://github.com/Mohammed-Lr/Call-Center-Monitoring-System.git
cd call-center-monitoring-system
```

## Step 2: Create Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Install dlib (for face_recognition)

**Windows:**
```bash
pip install cmake
pip install dlib
```

**Linux:**
```bash
sudo apt-get install cmake
sudo apt-get install libboost-all-dev
pip install dlib
```

**Mac:**
```bash
brew install cmake
brew install boost
pip install dlib
```

## Step 5: Download Models

Place the following files in the `models/` directory:
- `headsetmodel.pt`
- `pdfinal.pt`
- `face_encodings.pickle`

## Step 6: Prepare Face Dataset

1. Create directories for each person in `data/pickle_dataset/`
2. Add face images for each person
3. Generate encodings:
```bash
python scripts/face_encodings_generator.py
```

## Step 7: Verify Installation

```bash
python examples/quick_start.py
```

## Troubleshooting

### OpenCV Issues
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

### face_recognition Issues
Install dlib with pre-built wheels:
```bash
pip install face_recognition_models
```

### CUDA/GPU Issues
For GPU support, install CUDA-enabled PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```