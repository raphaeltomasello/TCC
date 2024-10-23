import threading

import cv2
import mediapipe as mp
import numpy as np

# Inicializando os módulos Pose e Drawing do MediaPipe.
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Capturando o vídeo da webcam.
cap = cv2.VideoCapture(1)

# Verificando se a webcam foi aberta corretamente.
if not cap.isOpened():
    print("Erro ao abrir a webcam")
    exit()

# Obtendo a largura e altura do vídeo capturado.
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Definindo o codec e criando o objeto VideoWriter para MP4.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('vetores.mp4', fourcc, 20.0, (width * 2, height))

# Verificando se o VideoWriter foi aberto corretamente.
if not out.isOpened():
    print("Erro ao abrir o arquivo de vídeo para escrita")
    cap.release()
    exit()

# Variáveis globais para o frame e o controle de execução.
frame = None
running = True

def capture_frames():
    global frame, running
    while running:
        ret, new_frame = cap.read()
        if not ret:
            print("Erro na captura da webcam")
            running = False
            break
        frame = new_frame

capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()

# Inicializando o MediaPipe Pose.
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
    while running:
        if frame is None:
            continue

        # Conversão para o espaço de cor HSV.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Ajuste do canal de brilho (V) para lidar com iluminação excessiva.
        h, s, v = cv2.split(hsv)
        v = cv2.equalizeHist(v)  # Equalização de histograma no canal de brilho.
        hsv_equalized = cv2.merge((h, s, v))
        frame_equalized = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2BGR)

        # Convertendo o quadro para RGB.
        frame_rgb = cv2.cvtColor(frame_equalized, cv2.COLOR_BGR2RGB)

        # Processando o quadro com o MediaPipe.
        results = pose.process(frame_rgb)

        # Criar uma cópia do quadro para desenhar os landmarks.
        annotated_frame = frame_equalized.copy()
        vectors_frame = np.zeros_like(frame_equalized)

        # Verificando se landmarks foram detectados.
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                landmark_list=results.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            mp_drawing.draw_landmarks(
                vectors_frame,
                landmark_list=results.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

        # Concatenar as duas imagens horizontalmente.
        combined_frame = np.hstack((annotated_frame, vectors_frame))

        # Gravar o quadro com vetores no arquivo de vídeo.
        out.write(combined_frame)

        # Exibindo o quadro combinado.
        cv2.imshow('Análise de Movimentos - Tênis', combined_frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            running = False
            break

# Liberando os recursos.
running = False
capture_thread.join()
cap.release()
out.release()
cv2.destroyAllWindows()