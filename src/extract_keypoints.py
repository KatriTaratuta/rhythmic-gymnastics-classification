import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

# Путь к папке с видео
DATA_PATH = 'D:/rhythmic-gymnastics-classification/data/raw_videos'
OUTPUT_PATH = 'D:/rhythmic-gymnastics-classification/data/keypoints_sequences'
ANNOTATIONS_PATH = 'D:/rhythmic-gymnastics-classification/data/annotations'  # Путь к аннотациям
SEQUENCE_LENGTH = 30
NUM_KEYPOINTS = 33
NUM_COORDINATES = 2

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)

def extract_keypoints(frame):
    """Извлекает ключевые точки из кадра с помощью MediaPipe Pose."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        keypoints = np.array([[landmark.x, landmark.y] for landmark in landmarks]).flatten()
        return keypoints
    else:
        return None

def time_to_frame(time_value, fps):
    """Преобразует время в формате datetime.time в номер кадра."""
    total_seconds = time_value.hour * 3600 + time_value.minute * 60 + time_value.second
    return int(total_seconds * fps)

def process_video(video_path, class_name, output_dir, annotation_file=None):
    """Обрабатывает видео, извлекая ключевые точки и сохраняя их в виде .npy файлов, учитывая аннотации."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)  # Получаем частоту кадров видео

    if annotation_file:
        try:
            df = pd.read_excel(annotation_file)
            start_time = pd.to_datetime(df['start_frame'][0], format='%H:%M:%S').time()  # Преобразуем в datetime.time
            end_time = pd.to_datetime(df['end_frame'][0], format='%H:%M:%S').time()    # Преобразуем в datetime.time

            start_frame = time_to_frame(start_time, fps)  # Преобразуем время в номер кадра
            end_frame = time_to_frame(end_time, fps)      # Преобразуем время в номер кадра

            print(f"Using annotation: start_frame={start_frame}, end_frame={end_frame}")
        except Exception as e:
            print(f"Error reading annotation file {annotation_file}: {e}")
            return
    else:
        start_frame = 0
        end_frame = float('inf')  # Бесконечность

    frame_count = 0
    keypoints_sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count < start_frame:
            continue  # Пропускаем кадры до начала нужного элемента
        if frame_count > end_frame:
            break  # Заканчиваем обработку после нужного элемента

        keypoints = extract_keypoints(frame)
        if keypoints is not None:
            keypoints_sequence.append(keypoints)
        else:
            print(f"No pose detected in frame {frame_count} of {video_path}")
            keypoints_sequence.append(np.zeros(NUM_KEYPOINTS * NUM_COORDINATES))  # Заполняем нулями

        if len(keypoints_sequence) == SEQUENCE_LENGTH:
            output_filename = os.path.join(output_dir, f"{class_name}_{os.path.basename(video_path).split('.')[0]}_{start_frame}_{frame_count}.npy")
            np.save(output_filename, np.array(keypoints_sequence))
            keypoints_sequence = []

    # Сохраняем оставшиеся кадры, если их достаточно
    if len(keypoints_sequence) > 0:
        output_filename = os.path.join(output_dir, f"{class_name}_{os.path.basename(video_path).split('.')[0]}_{start_frame}_partial_{frame_count}.npy")
        np.save(output_filename, np.array(keypoints_sequence))

    cap.release()
    print(f"Processed {video_path} and saved keypoints to {output_dir}")

def main():
    """Главная функция для обработки видеофайлов."""
    classes = ['balance', 'jump', 'rotation1', 'rotation2', 'rotation3', 'rotation4.1', 'rotation4.2', 'rotation4.3', 'rotation4.4']

    # Создаем папки для каждого класса, если их нет
    for class_name in classes:
        output_dir = os.path.join(OUTPUT_PATH, class_name)
        os.makedirs(output_dir, exist_ok=True)

    for class_name in classes:
        video_filename = f"{class_name}.mov"
        video_path = os.path.join(DATA_PATH, video_filename)
        annotation_file = None

        # Проверяем, есть ли аннотации для этого класса
        if class_name.startswith('rotation4'):
            annotation_file = os.path.join(ANNOTATIONS_PATH, f"{class_name}.xls")  # Изменено на .xls

        # Проверяем, существует ли файл видео
        if os.path.exists(video_path):
            output_dir = os.path.join(OUTPUT_PATH, class_name)
            process_video(video_path, class_name, output_dir, annotation_file)
        else:
            print(f"Video file not found: {video_path}")

if __name__ == "__main__":
    main()