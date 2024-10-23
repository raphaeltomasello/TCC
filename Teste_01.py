import cv2
import mediapipe as mp

# Inicializando os módulos Pose e Drawing do MediaPipe.
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Capturando o vídeo da webcam.
cap = cv2.VideoCapture(1)

# Definindo a resolução desejada (por exemplo, 1280x720).
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("Erro ao abrir a webcam")
    exit()

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Erro na captura da webcam")
            break

        # Conversão para RGB.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processamento do quadro com o MediaPipe.
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            annotated_frame = frame.copy()
            mp_drawing.draw_landmarks(
                annotated_frame,
                landmark_list=results.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            # Iterar sobre todos os landmarks e imprimir as coordenadas.
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                height, width, _ = frame.shape
                x, y = int(landmark.x * width), int(landmark.y * height)
                
                # Obter o nome da parte do corpo utilizando o índice.
                body_part = mp_pose.PoseLandmark(idx).name
                
                # Verificar se o landmark está visível (visibilidade > 0.5).
                if landmark.visibility > 0.5:
                    print(f'{body_part}: Coordenadas ({x}, {y})')

            cv2.imshow('Análise de Movimentos - Tênis', annotated_frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()