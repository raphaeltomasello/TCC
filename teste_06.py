import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

import cv2
import mediapipe as mp
import numpy as np
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm

# Inicializando os módulos Pose e Drawing do MediaPipe.
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Definindo a altura padrão
STANDARD_HEIGHT = 360

# Função para redimensionar as imagens mantendo a proporção
def resize_with_aspect_ratio(image, height=STANDARD_HEIGHT):
    aspect_ratio = image.shape[1] / image.shape[0]
    new_width = int(height * aspect_ratio)
    return cv2.resize(image, (new_width, height))

# Função para calcular a diferença entre landmarks
def calculate_landmark_difference(landmarks1, landmarks2):
    if landmarks1 is None or landmarks2 is None:
        return None
    
    differences = []
    for lm1, lm2 in zip(landmarks1.landmark, landmarks2.landmark):
        diff = np.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2 + (lm1.z - lm2.z)**2)
        differences.append(diff)
    
    return np.mean(differences)

# Função para detectar e desenhar a bola de tênis
def detect_tennis_ball(frame):
    # Converter a imagem para o espaço de cores HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Definir o intervalo de cor para a bola de tênis (amarelo)
    '''lower_yellow = np.array([25, 70, 120])
    upper_yellow = np.array([35, 255, 255])'''

    # Ajuste os valores de acordo com a aparência da bola no vídeo
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    # Criar uma máscara para a cor amarela
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Aplicar operações morfológicas para remover ruídos
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Encontrar contornos na máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Verificar se algum contorno foi encontrado
    if contours:
        # Encontrar o contorno com a maior área
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

        # Desenhar o círculo ao redor da bola de tênis
        if radius > 10:  # Ajuste o tamanho mínimo da bola detectada
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.putText(frame, "Bola de Tenis", (int(x) - 10, int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            return (x, y)

    return None

# Função para processar um vídeo
def process_video(video_path):
    video = cv2.VideoCapture(video_path)
    poses = []
    frames = []
    ball_positions = []
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = 2  # Processa um a cada 2 frames

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0) as pose:
        for i in tqdm(range(0, total_frames, frame_skip), desc=f"Processando vídeo {video_path}"):
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            if not ret:
                break

            frame_resized = resize_with_aspect_ratio(frame)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                poses.append(results.pose_landmarks)
                frames.append(frame_resized)

                # Detectar e salvar a posição da bola
                ball_position = detect_tennis_ball(frame_resized.copy())
                ball_positions.append(ball_position)

    video.release()
    return poses, frames, ball_positions

# Processar os dois vídeos
print("Processando o primeiro vídeo...")
poses1, frames1, ball_positions1 = process_video('01.mp4')
print(f"Primeiro vídeo processado. Total de poses capturadas: {len(poses1)}")

print("Processando o segundo vídeo...")
poses2, frames2, ball_positions2 = process_video('02.mp4')
print(f"Segundo vídeo processado. Total de poses capturadas: {len(poses2)}")

# Lista para armazenar os frames processados
processed_frames = []

# Comparando os dois vídeos
for i in range(min(len(poses1), len(poses2))):
    frame1 = frames1[i].copy()
    frame2 = frames2[i].copy()
    comparison_frame = np.zeros((STANDARD_HEIGHT, STANDARD_HEIGHT, 3), dtype=np.uint8)

    # Desenhar landmarks e calcular diferenças
    mp_drawing.draw_landmarks(
        frame1,
        landmark_list=poses1[i],
        connections=mp_pose.POSE_CONNECTIONS
    )
    mp_drawing.draw_landmarks(
        frame2,
        landmark_list=poses2[i],
        connections=mp_pose.POSE_CONNECTIONS
    )

    # Detectar e desenhar a bola de tênis
    ball_pos1 = ball_positions1[i]
    ball_pos2 = ball_positions2[i]

    # Calcular a diferença de posição da bola
    if ball_pos1 and ball_pos2:
        ball_difference = np.sqrt((ball_pos1[0] - ball_pos2[0])**2 + (ball_pos1[1] - ball_pos2[1])**2)
        cv2.putText(comparison_frame, f"Dif. Bola: {ball_difference:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Calcular a diferença entre os landmarks
    difference = calculate_landmark_difference(poses1[i], poses2[i])

    # Desenhar feedback visual no quadro de comparação
    if difference is not None:
        color = (0, 255, 0) if difference < 0.1 else (0, 165, 255) if difference < 0.2 else (0, 0, 255)
        cv2.putText(comparison_frame, f"Dif. Landmarks: {difference:.4f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Concatenar as três imagens horizontalmente.
    combined_frame = np.hstack((frame1, frame2, comparison_frame))

    # Adicionar o frame processado à lista
    processed_frames.append(combined_frame)

    # Exibindo o quadro combinado.
    cv2.imshow('Comparação de Movimentos', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# Criando o vídeo com moviepy
print("Salvando o vídeo...")
clip = ImageSequenceClip(processed_frames, fps=20)
clip.write_videofile("comparacao_movimentos.mp4", codec='libx264')
print("Vídeo salvo com sucesso como 'comparacao_movimentos.mp4'")