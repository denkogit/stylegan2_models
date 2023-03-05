import os 
from image_aligner.face_alignment import image_align
from image_aligner.landmarks_detector import LandmarksDetector


landmarks_model_path = ""
RAW_IMAGES_DIR = ""
ALIGNED_IMAGES_DIR = ""


landmarks_detector = LandmarksDetector(landmarks_model_path)
for img_name in os.listdir(RAW_IMAGES_DIR):
    raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
    for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
        face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
        aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)

        image_align(raw_img_path, aligned_face_path, face_landmarks)