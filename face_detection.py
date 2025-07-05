import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN

# Function to detect and draw faces with confidence filtering
def draw_faces_with_mtcnn(image, confidence_threshold=0.90, min_face_size=30):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    faces = detector.detect_faces(rgb)
    
    filtered_faces = []
    for face in faces:
        x, y, w, h = face['box']
        confidence = face['confidence']
        if confidence >= confidence_threshold and w >= min_face_size and h >= min_face_size:
            filtered_faces.append(face)
            cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    return rgb, len(filtered_faces)

# Load image paths
image1_path = "C:/Users/maryam/Desktop/LVL 4, Semester 2/Pattern Recognition/Project/team.jpg"
image2_path = "C:/Users/maryam/Desktop/LVL 4, Semester 2/Pattern Recognition/Project/team2.jpg"

# Read images
img1 = cv2.imread(image1_path)
img2 = cv2.imread(image2_path)

# Check loading status
if img1 is None or img2 is None:
    print("One or both images could not be loaded.")
else:
    print("Images loaded successfully.")

    # Detect and annotate faces
    img1_with_faces, count1 = draw_faces_with_mtcnn(img1)
    img2_with_faces, count2 = draw_faces_with_mtcnn(img2)

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    axes[0].imshow(img1_with_faces)
    axes[0].set_title(f"Faces in image1: {count1}")
    axes[0].axis('off')

    axes[1].imshow(img2_with_faces)
    axes[1].set_title(f"Faces in image2: {count2}")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
