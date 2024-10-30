import cv2   
import numpy as np         # Numpy primeiro para carregar dependências matemáticas
import pandas as pd        # Pandas não afeta as demais, então vem após numpy
import mediapipe as mp     # MediaPipe após OpenCV
import tempfile            # Agora outras bibliotecas da biblioteca padrão
import os
import math
import streamlit as st     # Streamlit como último para garantir que as demais estejam carregadas

# Configuração do MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Funções auxiliares para cálculo de ângulo e RULA Score
def calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
    degree = int(180 / m.pi) * theta
    return degree

def calculate_rula_score_tronco(angle):
    if angle <= 5:
        return 1
    elif angle <= 20:
        return 2
    elif angle <= 60:
        return 3
    return 4

def calculate_rula_score_pescoco(angle):
    if angle <= 10:
        return 1
    elif angle <= 20:
        return 2
    return 3

def calculate_rula_score_antebraco(angle):
    if 60 < angle < 100:
        return 1
    return 2

def calculate_rula_score_braco(angle):
    if angle <= 20:
        return 1
    elif angle <= 45:
        return 2
    elif angle <= 90:
        return 3
    return 4

def convert_seconds_to_hhmmss(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

st.title("🎒 Metodo Rula - Ergonomia")

# Carregar vídeo
uploaded_video = st.file_uploader("Escolha um vídeo", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    # Salvar vídeo de entrada em arquivo temporário e fechar o arquivo após escrever
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    tfile.close()

    # Ler o vídeo usando OpenCV
    video = cv2.VideoCapture(tfile.name)
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    data = []

    with mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5) as holistic:
        frame_index = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                height, width, _ = frame.shape

                # Verificar a visibilidade dos lados direito e esquerdo
                right_visibility = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER].visibility
                left_visibility = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER].visibility

                

                if right_visibility > left_visibility:
                    # Cálculos para o lado direito
                    shoulder = mp_holistic.PoseLandmark.RIGHT_SHOULDER
                    hip = mp_holistic.PoseLandmark.RIGHT_HIP
                    elbow = mp_holistic.PoseLandmark.RIGHT_ELBOW
                    wrist = mp_holistic.PoseLandmark.RIGHT_WRIST
                    ear = mp_holistic.PoseLandmark.RIGHT_EAR
                    side = "Direito"
                else:
                    # Cálculos para o lado esquerdo
                    shoulder = mp_holistic.PoseLandmark.LEFT_SHOULDER
                    hip = mp_holistic.PoseLandmark.LEFT_HIP
                    elbow = mp_holistic.PoseLandmark.LEFT_ELBOW
                    wrist = mp_holistic.PoseLandmark.LEFT_WRIST
                    ear = mp_holistic.PoseLandmark.LEFT_EAR
                    side = "Esquerdo"

                # Coordenadas dos pontos relevantes
                shoulder_coord = (int(landmarks[shoulder].x * width), int(landmarks[shoulder].y * height))
                hip_coord = (int(landmarks[hip].x * width), int(landmarks[hip].y * height))
                elbow_coord = (int(landmarks[elbow].x * width), int(landmarks[elbow].y * height))
                wrist_coord = (int(landmarks[wrist].x * width), int(landmarks[wrist].y * height))
                ear_coord = (int(landmarks[ear].x * width), int(landmarks[ear].y * height))

                # Cálculo dos ângulos e RULA scores
                angle_tronco = calculate_angle(shoulder_coord, hip_coord, [hip_coord[0], hip_coord[1] - 1])
                rula_score_tronco = calculate_rula_score_tronco(angle_tronco)

                neck_angle = calculate_angle(shoulder_coord, ear_coord, hip_coord)
                rula_score_pescoco = calculate_rula_score_pescoco(neck_angle)

                angle_antebraco = calculate_angle(shoulder_coord, elbow_coord, wrist_coord)
                rula_score_antebraco = calculate_rula_score_antebraco(angle_antebraco)

                angle_braco = calculate_angle(hip_coord, shoulder_coord, wrist_coord)
                rula_score_braco = calculate_rula_score_braco(angle_braco)
                
                # Exibir ângulos na parte superior do frame
                text_y_start_top = int(height * 0.05)  # Começo do texto para ângulos
                line_spacing = int(height * 0.05)  # Espaçamento entre linhas

                cv2.putText(frame, f'Ang. Tronco {side}: {int(angle_tronco)}', (int(width * 0.02), text_y_start_top),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Ang. Pescoco {side}: {int(neck_angle)}', (int(width * 0.02), text_y_start_top + line_spacing),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Ang. Antebraco {side}: {int(angle_antebraco)}', (int(width * 0.02), text_y_start_top + 2 * line_spacing),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Ang. Braco {side}: {int(angle_braco)}', (int(width * 0.02), text_y_start_top + 3 * line_spacing),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Exibir RULA scores na parte inferior do frame
                text_y_start_bottom = int(height * 0.8)  # Início dos RULA scores na parte inferior
                cv2.putText(frame, f'Rula Score Tronco: {rula_score_tronco}', (int(width * 0.02), text_y_start_bottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Rula Score Pescoco: {rula_score_pescoco}', (int(width * 0.02), text_y_start_bottom + line_spacing),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Rula Score Antebraco: {rula_score_antebraco}', (int(width * 0.02), text_y_start_bottom + 2 * line_spacing),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Rula Score Braco: {rula_score_braco}', (int(width * 0.02), text_y_start_bottom + 3 * line_spacing),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                cv2.circle(frame, tuple(shoulder_coord), 5, (0, 255, 0), -1)
                cv2.circle(frame, tuple(hip_coord), 5, (0, 255, 0), -1)
                cv2.line(frame, tuple(hip_coord), tuple(shoulder_coord), (0, 255, 0), 2)
                cv2.circle(frame, (int(ear_coord[0]), int(ear_coord[1])), 5, (0, 255, 0), -1)
                cv2.line(frame, (int(ear_coord[0]), int(ear_coord[1])), (int(shoulder_coord[0]), int(shoulder_coord[1])), (0, 255, 0), 2)
                cv2.circle(frame, tuple(elbow_coord), 5, (0, 255, 0), -1)
                cv2.circle(frame, tuple(wrist_coord), 5, (0, 255, 0), -1)
                cv2.line(frame, tuple(shoulder_coord), tuple(elbow_coord), (0, 255, 0), 2)
                cv2.line(frame, tuple(elbow_coord), tuple(wrist_coord), (0, 255, 0), 2)


                # Armazenar os dados no dicionário
                current_time = frame_index / fps
                current_time_hhmmss = convert_seconds_to_hhmmss(current_time)

                data.append({
                    'Tempo (s)': current_time_hhmmss,
                    f'Ângulo Tronco {side}': angle_tronco,
                    'Rula Score Tronco': rula_score_tronco,
                    f'Ângulo Pescoco {side}': neck_angle,
                    'Rula Score Pescoco': rula_score_pescoco,
                    f'Ângulo Antebraco {side}': angle_antebraco,
                    'Rula Score Antebraco': rula_score_antebraco,
                    f'Ângulo Braco {side}': angle_braco,
                    'Rula Score Braco': rula_score_braco
                })

            out.write(frame)
            frame_index += 1

    video.release()
    out.release()

        # Salvar dados em planilha
    df = pd.DataFrame(data)
    excel_path = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx").name
    df.to_excel(excel_path, index=False)

        # Agrupar dados por segundo e calcular médias
    grouped_df = df.groupby('Tempo (s)').median().reset_index()
    grouped_excel_path = tempfile.NamedTemporaryFile(delete=False, suffix="_grouped.xlsx").name
    grouped_df.to_excel(grouped_excel_path, index=False)


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

    with open(grouped_excel_path, "rb") as f:
        st.download_button(
            label="Baixar planilha de ângulos por segundo",
            data=f, file_name="angulo_tronco_por_segundo.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    # Remover arquivos temporários após o uso
    try:
        os.remove(tfile.name)
    except PermissionError:
        pass  # Ignorar erro se o arquivo ainda estiver em uso
    os.remove(temp_video_path)
    os.remove(excel_path)
    os.remove(grouped_excel_path)

st.write("Desenvolvido por [Yuri Rocha](https://www.linkedin.com/in/yuri-rudimar/) com apoio de [Vanessa Nappi](http://lattes.cnpq.br/1442468348335571) para a disciplina de Ergonomia.
Departamento de Engenharia de Produção e Sistema, UDESC-CCT (Joinville-SC)")
