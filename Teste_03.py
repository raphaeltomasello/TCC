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

# Função para processar o vídeo de referência
def process_reference_video(video_path):
    reference_video = cv2.VideoCapture(video_path)
    reference_poses = []
    reference_frames = []
    
    total_frames = int(reference_video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = 2  # Processa um a cada 2 frames

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0) as pose:
        for i in tqdm(range(0, total_frames, frame_skip), desc="Processando vídeo de referência"):
            reference_video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = reference_video.read()
            if not ret:
                break

            frame_resized = resize_with_aspect_ratio(frame)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                reference_poses.append(results.pose_landmarks)
                reference_frames.append(frame_resized)

    reference_video.release()
    return reference_poses, reference_frames

# Processar o vídeo de referência
print("Iniciando o processamento do vídeo de referência...")
reference_poses, reference_frames = process_reference_video('Roger Federer Serves from Back Perpsective in HD.mp4')
print(f"Vídeo de referência processado. Total de poses capturadas: {len(reference_poses)}")

# Capturando o vídeo da webcam.
cap = cv2.VideoCapture(0)

# Verificando se a webcam foi aberta corretamente.
if not cap.isOpened():
    print("Erro ao abrir a webcam")
    exit()

# Lista para armazenar os frames processados
processed_frames = []

try:
    # Inicializando o MediaPipe Pose para a webcam.
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0) as pose:
        reference_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Falha ao capturar frame da webcam")
                break

            # Redimensionar o frame
            frame_resized = resize_with_aspect_ratio(frame)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            # Processar o frame da webcam
            results_webcam = pose.process(frame_rgb)

            # Criar frames para desenho
            annotated_frame = frame_resized.copy()
            comparison_frame = np.zeros((STANDARD_HEIGHT, STANDARD_HEIGHT, 3), dtype=np.uint8)

            # Obter o frame de referência atual
            reference_frame = reference_frames[reference_index].copy()

            # Desenhar landmarks e calcular diferenças
            if results_webcam.pose_landmarks and reference_index < len(reference_poses):
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    landmark_list=results_webcam.pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS
                )
                mp_drawing.draw_landmarks(
                    reference_frame,
                    landmark_list=reference_poses[reference_index],
                    connections=mp_pose.POSE_CONNECTIONS
                )

                # Calcular a diferença entre os landmarks
                difference = calculate_landmark_difference(results_webcam.pose_landmarks, reference_poses[reference_index])

                # Desenhar feedback visual no quadro de comparação
                if difference is not None:
                    color = (0, 255, 0) if difference < 0.1 else (0, 165, 255) if difference < 0.2 else (0, 0, 255)
                    cv2.putText(comparison_frame, f"Diferenca: {difference:.4f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # Desenhar landmarks com cor baseada na diferença
                    mp_drawing.draw_landmarks(
                        comparison_frame,
                        landmark_list=results_webcam.pose_landmarks,
                        connections=mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=color, thickness=2)
                    )

                reference_index = (reference_index + 1) % len(reference_poses)

            # Concatenar as três imagens horizontalmente.
            combined_frame = np.hstack((reference_frame, annotated_frame, comparison_frame))

            # Adicionar o frame processado à lista
            processed_frames.append(combined_frame)

            # Exibindo o quadro combinado.
            cv2.imshow('Comparação de Movimentos', combined_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    # Liberando os recursos.
    cap.release()
    cv2.destroyAllWindows()

# Criando o vídeo com moviepy
print("Salvando o vídeo...")
clip = ImageSequenceClip(processed_frames, fps=20)
clip.write_videofile("comparacao_movimentos.mp4", codec='libx264')
print("Vídeo salvo com sucesso como 'comparacao_movimentos.mp4'")   