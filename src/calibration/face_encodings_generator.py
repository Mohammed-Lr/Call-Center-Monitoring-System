import face_recognition
import pickle
from pathlib import Path

def generate_encodings(dataset_path, output_path):
    known_face_encodings = []
    known_face_names = []

    for person_dir in Path(dataset_path).iterdir():
        if person_dir.is_dir():
            person_name = person_dir.name
            print(f"Processing images for {person_name}")

            for image_path in person_dir.glob("*"):
                if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    try:
                        image = face_recognition.load_image_file(str(image_path))
                        encodings = face_recognition.face_encodings(image)

                        if encodings:
                            known_face_encodings.append(encodings[0])
                            known_face_names.append(person_name)
                            print(f"Successfully encoded {image_path.name}")
                        else:
                            print(f"No face found in {image_path.name}")
                    except Exception as e:
                        print(f"Error processing {image_path}: {str(e)}")

    data = {
        "encodings": known_face_encodings,
        "names": known_face_names
    }

    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"\nEncoding complete! Processed {len(known_face_encodings)} faces.")
    return data