import os
import pandas as pd
from moviepy.editor import VideoFileClip

print("Текущая рабочая директория:", os.getcwd())

# Используем абсолютные пути
base_dir = r"D:\rhythmic-gymnastics-classification"
raw_videos_dir = os.path.join(base_dir, 'data', 'raw_videos')
annotations_dir = os.path.join(base_dir, 'data', 'annotations')
processed_frames_dir = os.path.join(base_dir, 'data', 'processed_videos', 'frames')

# Создание папки для обработанных кадров
os.makedirs(processed_frames_dir, exist_ok=True)

# Классы гимнастических элементов
classes = {
    'balance': 'balance.mov',
    'jump': 'jump.mov',
    'rotation': [
        'rotation1.mov', 'rotation2.mov', 'rotation3.mov',
        'rotation4.1.mov', 'rotation4.2.mov', 'rotation4.3.mov', 'rotation4.4.mov'
    ]
}

# Обработчик видео
def process_video(video_path, output_dir, start_time=None, end_time=None):
    if not os.path.exists(video_path):
        print(f"Видеофайл не найден: {video_path}")
        return

    try:
        clip = VideoFileClip(video_path)
    except Exception as e:
        print(f"Ошибка при открытии видео {video_path}: {e}")
        return

    start_time = start_time or 0
    end_time = end_time or clip.duration

    # Итерация по времени, сохраняем кадры
    for t in range(int(start_time), int(end_time)):
        frame_filename = os.path.join(output_dir, f'frame_{t}.jpg')
        try:
            clip.save_frame(frame_filename, t=t)
        except Exception as e:
            print(f"Ошибка при сохранении кадра {frame_filename}: {e}")

# Обработка каждого класса
for element, video_names in classes.items():
    if isinstance(video_names, str):
        video_names = [video_names]

    for video_name in video_names:
        video_path = os.path.join(raw_videos_dir, video_name)

        # Отладочный вывод
        print(f"Проверяю файл: {video_path}")

        if not os.path.exists(video_path):
            print(f"Файл не найден: {video_path}")
            continue

        output_dir = os.path.join(processed_frames_dir, element)
        os.makedirs(output_dir, exist_ok=True)

        # Проверяем, есть ли разметка для видео
        annotation_file = os.path.join(annotations_dir, f"{video_name.split('.')[0]}.xlsx")
        if os.path.exists(annotation_file):
            df = pd.read_excel(annotation_file)
            for _, row in df.iterrows():
                process_video(video_path, output_dir, row['Start Time'], row['End Time'])
        else:
            process_video(video_path, output_dir)

print('Data preparation completed.')