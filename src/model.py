import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Параметры
SEQUENCE_LENGTH = 30
NUM_KEYPOINTS = 33
NUM_COORDINATES = 2
NUM_CLASSES = 3  # 3 класса: balance, jump, rotation
BATCH_SIZE = 32
EPOCHS = 30
DATA_DIR = r'D:\rhythmic-gymnastics-classification\data\keypoints_sequences'
MODEL_SAVE_PATH = r'D:\rhythmic-gymnastics-classification\models\best_model.pth'

# Класс для загрузки данных
class GymnasticsDataset(Dataset):
    def __init__(self, data_dir, sequence_length, num_keypoints, num_coordinates):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.num_keypoints = num_keypoints
        self.num_coordinates = num_coordinates
        self.X, self.y = self.load_data()

    def load_data(self):
        X, y = [], []
        class_names = os.listdir(self.data_dir)

        class_indices = {'balance': 0, 'jump': 1, 'rotation': 2}

        for class_name in class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            for sequence_file in os.listdir(class_dir):
                sequence_path = os.path.join(class_dir, sequence_file)
                sequence = np.load(sequence_path)

                if sequence.shape[1] != self.num_keypoints * self.num_coordinates:
                    continue

                sequence = sequence[:self.sequence_length] if len(sequence) > self.sequence_length else np.pad(
                    sequence, ((0, self.sequence_length - len(sequence)), (0, 0)), mode='constant')

                X.append(sequence.astype('float32') / 1000.0)

                if 'rotation' in class_name:
                    y.append(class_indices['rotation'])
                else:
                    y.append(class_indices.get(class_name, -1))

        X = np.array(X)
        y = np.array(y)

        return X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


class HybridModel(nn.Module):
    def __init__(self, sequence_length, num_keypoints, num_coordinates, num_classes):
        super(HybridModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=num_keypoints * num_coordinates, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, num_classes)  # Умножаем на 2 для двунаправленного LSTM

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, -1).permute(0, 2, 1)  
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        x = self.fc(lstm_out[:, -1, :])
        return x


# Функции обучения и тестирования
def train_model(model, train_loader, optimizer, loss_fn):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# Функция для сохранения модели
def save_model(model, path):
    torch.save(model.state_dict(), path)


if __name__ == '__main__':
    dataset = GymnasticsDataset(DATA_DIR, SEQUENCE_LENGTH, NUM_KEYPOINTS, NUM_COORDINATES)

    # Разделение на обучающую и тестовую выборку
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # Убедитесь, что индексы не выходят за пределы
    indices = np.random.permutation(len(dataset))
    train_indices, test_indices = indices[:train_size], indices[train_size:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = HybridModel(SEQUENCE_LENGTH, NUM_KEYPOINTS, NUM_COORDINATES, NUM_CLASSES)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        train_loss = train_model(model, train_loader, optimizer, loss_fn)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {train_loss:.4f}")

    accuracy = evaluate_model(model, test_loader)
    print(f"Test Accuracy: {accuracy:.2f}%")

    save_model(model, MODEL_SAVE_PATH)
    print(f"Model saved at {MODEL_SAVE_PATH}")