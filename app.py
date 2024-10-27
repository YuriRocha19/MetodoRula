import numpy as np         # Numpy primeiro para carregar dependências matemáticas
import pandas as pd        # Pandas não afeta as demais, então vem após numpy
import cv2                 # OpenCV antes do MediaPipe para minimizar conflitos
import mediapipe as mp     # MediaPipe após OpenCV
import tempfile            # Agora outras bibliotecas da biblioteca padrão
import os
import math
import streamlit as st     # Streamlit como último para garantir que as demais estejam carregadas

# Configuração do MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Função para calcular o ângulo entre dois pontos em relação ao eixo vertical
def calculate_trunk_angle(shoulder_mid, hip_mid):
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    angle_radians = math.atan2(dy, dx)
    angle_degrees = abs(math.degrees(angle_radians))
    return angle_degrees

# Função para processar frame do vídeo com MediaPipe e calcular o ângulo
def process_frame_with_landmarks(frame, holistic):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    if results.pose_landmarks:
        height, width, _ = frame.shape
        
        # Coordenadas dos ombros e quadris
        left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
        left_hip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP]
        
        # Calcular pontos médios dos ombros e quadris
        shoulder_mid = (
            int((left_shoulder.x + right_shoulder.x) / 2 * width),
            int((left_shoulder.y + right_shoulder.y) / 2 * height)
        )
        hip_mid = (
            int((left_hip.x + right_hip.x) / 2 * width),
            int((left_hip.y + right_hip.y) / 2 * height)
        )
        
        # Desenhar pontos dos ombros e quadris
        cv2.circle(frame, shoulder_mid, 5, (255, 0, 0), -1)
        cv2.circle(frame, hip_mid, 5, (255, 0, 0), -1)
        
        # Calcular o ângulo do tronco
        trunk_angle = calculate_trunk_angle(shoulder_mid, hip_mid)
        
        # Exibir o ângulo na tela
        cv2.putText(
            frame, f"Trunk Angle: {trunk_angle:.2f} degrees", 
            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        
        # Desenhar linha entre ombros e quadris
        cv2.line(frame, shoulder_mid, hip_mid, (0, 255, 0), 2)

        return frame, trunk_angle  # Retorna também o ângulo
    return frame, None

st.title("Ergonomia com Processamento de Vídeo usando MediaPipe")

# Carregar vídeo
uploaded_video = st.file_uploader("Escolha um vídeo", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    # Salvar vídeo de entrada em arquivo temporário e fechar o arquivo após escrever
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    tfile.close()  # Fechar o arquivo temporário após o upload
    
    # Ler o vídeo usando OpenCV
    video = cv2.VideoCapture(tfile.name)

    # Configurar saída de vídeo em um arquivo temporário
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    stframe = st.empty()  # Placeholder para exibir os frames

    # Lista para armazenar dados dos ângulos
    angle_data = []

    # Processo de vídeo com contexto do MediaPipe
    with mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5) as holistic:
        frame_index = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # Processar frame e calcular o ângulo do tronco
            processed_frame, trunk_angle = process_frame_with_landmarks(frame, holistic)

            # Salvar dados do ângulo se calculado
            if trunk_angle is not None:
                time_in_seconds = frame_index / fps
                angle_data.append({"Time (s)": time_in_seconds, "Trunk Angle (degrees)": trunk_angle})
            
            # Incrementar o índice do frame
            frame_index += 1

            # Escrever frame processado no vídeo de saída
            out.write(processed_frame)

            # Exibir frame
            stframe.image(processed_frame, channels="BGR")

    # Fechar streams de vídeo
    video.release()
    out.release()

    # Criar DataFrame e exportar para Excel
    df = pd.DataFrame(angle_data)
    excel_path = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx").name
    df.to_excel(excel_path, index=False)

    # Exibir botão de download para o vídeo processado
    with open(temp_video_path, "rb") as f:
        st.download_button(
            label="Baixar vídeo processado",
            data=f,
            file_name="video_processado.mp4",
            mime="video/mp4"
        )

    # Exibir botão de download para o arquivo Excel
    with open(excel_path, "rb") as f:
        st.download_button(
            label="Baixar planilha de ângulos",
            data=f,
            file_name="angulo_tronco.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # Remover arquivos temporários após o uso
    try:
        os.remove(tfile.name)
    except PermissionError:
        pass  # Ignorar erro se o arquivo ainda estiver em uso
    os.remove(temp_video_path)
    os.remove(excel_path)

st.write("Para mais informações, visite o [LinkedIn do autor](https://www.linkedin.com/in/yuri-rudimar/)")
