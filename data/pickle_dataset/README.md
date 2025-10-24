# Face Dataset for Encodings

This directory contains face images for generating face encodings.

## Directory Structure

```
pickle_dataset/
├── person1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── image3.jpg
├── person2/
│   ├── image1.jpg
│   └── image2.jpg
└── ...
```

## Adding New Faces

1. Create a directory with the person's name
2. Add 3-5 clear face images of the person
3. Ensure images are well-lit and face is clearly visible
4. Run the encoding generator:
   ```bash
   python scripts/face_encodings_generator.py
   ```

## Image Requirements

- Format: JPG, JPEG, or PNG
- Face should be clearly visible
- Good lighting conditions
- Multiple angles recommended
- Minimum 3 images per person

## Generating Encodings

After adding images, run:
```bash
python src/calibration/face_encodings_generator.py
```

This will create/update `models/face_encodings.pickle`