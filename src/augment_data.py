import os
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageEnhance

# Пути к папкам
base_dir = r"D:\rhythmic-gymnastics-classification"
processed_frames_dir = os.path.join(base_dir, 'data', 'processed_videos', 'frames')
enriched_frames_dir = os.path.join(base_dir, 'data', 'processed_videos', 'enriched_frames')
augmented_frames_dir = os.path.join(base_dir, 'data', 'processed_videos', 'augmented_frames')

os.makedirs(enriched_frames_dir, exist_ok=True)
os.makedirs(augmented_frames_dir, exist_ok=True)

# Инициализация MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Обогащение данных (добавление ключевых точек)
def enrich_frame(image_path, output_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

    cv2.imwrite(output_path, image)

# Аугментация данных (создание вариаций)
def augment_frame(image_path, output_dir):
    image = Image.open(image_path)

    for angle in [0, 15, -15]:
        rotated = image.rotate(angle)
        rotated.save(os.path.join(output_dir, f"rotated_{angle}.jpg"))

    enhancer = ImageEnhance.Brightness(image)
    enhancer.enhance(1.5).save(os.path.join(output_dir, "bright.jpg"))
    enhancer.enhance(0.5).save(os.path.join(output_dir, "dark.jpg"))

    enhancer = ImageEnhance.Contrast(image)
    enhancer.enhance(1.5).save(os.path.join(output_dir, "high_contrast.jpg"))
    enhancer.enhance(0.5).save(os.path.join(output_dir, "low_contrast.jpg"))

# Обработка кадров
for class_folder in os.listdir(processed_frames_dir):
    class_path = os.path.join(processed_frames_dir, class_folder)

    enriched_class_path = os.path.join(enriched_frames_dir, class_folder)
    augmented_class_path = os.path.join(augmented_frames_dir, class_folder)

    os.makedirs(enriched_class_path, exist_ok=True)
    os.makedirs(augmented_class_path, exist_ok=True)

    for frame in os.listdir(class_path):
        frame_path = os.path.join(class_path, frame)

        # Обогащение
        enriched_frame_path = os.path.join(enriched_class_path, frame)
        enrich_frame(frame_path, enriched_frame_path)

        # Аугментация
        augment_frame(frame_path, augmented_class_path)

print('Data enrichment and augmentation completed.')
