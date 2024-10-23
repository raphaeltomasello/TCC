import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Definindo a arquitetura da rede neural
class MovementComparisonModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MovementComparisonModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # CNN para extração de características
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        
        # LSTM para sequenciamento
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Camada de saída
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len, c, h, w = x.size()
        
        # Reshape para processar com CNN
        cnn_input = x.view(batch_size * seq_len, c, h, w)
        cnn_output = self.cnn(cnn_input)
        
        # Reshape de volta para sequência
        lstm_input = cnn_output.view(batch_size, seq_len, -1)
        
        # Processo LSTM
        lstm_out, _ = self.lstm(lstm_input)
        
        # Usar apenas o último output da sequência
        out = self.fc(lstm_out[:, -1, :])
        return out

# Função para extrair landmarks do MediaPipe
def extract_landmarks(frame, pose):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        return landmarks.flatten()
    return np.zeros(33 * 3)  # 33 landmarks, each with x, y, z

# Função para preparar sequência de frames
def prepare_sequence(video_path, pose, sequence_length=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break
        landmarks = extract_landmarks(frame, pose)
        frames.append(landmarks)
    cap.release()
    
    # Pad a sequência se for menor que sequence_length
    while len(frames) < sequence_length:
        frames.append(np.zeros_like(frames[0]))
    
    return np.array(frames)

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Hiperparâmetros
input_size = 33 * 3  # 33 landmarks, each with x, y, z
hidden_size = 64
num_layers = 2
output_size = 1  # Similaridade (0 a 1)
learning_rate = 0.001
num_epochs = 100
batch_size = 32

# Inicializar o modelo
model = MovementComparisonModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Função de treinamento
def train_model(model, train_data, train_labels):
    model.train()
    for epoch in range(num_epochs):
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Função para comparar movimentos
def compare_movements(model, reference_seq, webcam_seq):
    model.eval()
    with torch.no_grad():
        reference_tensor = torch.FloatTensor(reference_seq).unsqueeze(0)
        webcam_tensor = torch.FloatTensor(webcam_seq).unsqueeze(0)
        
        reference_output = model(reference_tensor)
        webcam_output = model(webcam_tensor)
        
        similarity = nn.functional.cosine_similarity(reference_output, webcam_output)
        return similarity.item()

# Exemplo de uso
reference_video_path = "caminho_para_video_referencia.mp4"
webcam_video_path = "caminho_para_video_webcam.mp4"

reference_seq = prepare_sequence(reference_video_path, pose)
webcam_seq = prepare_sequence(webcam_video_path, pose)

# Aqui você precisaria de dados de treinamento rotulados
# train_data = ...
# train_labels = ...
# train_model(model, train_data, train_labels)

similarity = compare_movements(model, reference_seq, webcam_seq)
print(f"Similaridade entre os movimentos: {similarity:.2f}")

# Liberar recursos
pose.close()