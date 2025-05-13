import os

# Путь к папке с обработанными кадрами
processed_frames_dir = 'data/processed_videos/frames'

# Проверяем существование папки
if not os.path.exists(processed_frames_dir):
    print('❌ Папка processed_videos/frames не существует.')
else:
    print('✅ Папка processed_videos/frames существует.')

# Проверяем наличие папок для каждого класса
classes = ['balance', 'jump', 'rotation']

for cls in classes:
    class_dir = os.path.join(processed_frames_dir, cls)
    if not os.path.exists(class_dir):
        print(f'❌ Папка для класса {cls} не существует.')
    else:
        print(f'✅ Папка для класса {cls} существует.')

        # Проверяем количество кадров
        frame_files = os.listdir(class_dir)
        if len(frame_files) == 0:
            print(f'⚠️ В папке {cls} нет кадров.')
        else:
            print(f'✅ В папке {cls} найдено {len(frame_files)} кадров.')

print('\nПроверка завершена.')
