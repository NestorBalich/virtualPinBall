#programa juego realidad virtual con IA 23-7-2024
#por Néstor Balicha
#TikTok https://www.tiktok.com/@nestorbalich
import cv2
import mediapipe as mp
import numpy as np
import time
import random
import winsound

# Parámetros configurables
BALL_SIZE = 30 #20
BALL_SPEED = 8 
SPEED_INCREMENT = 2
GAME_TIME = 30  # segundos
TARGET_COUNT = 10

# Inicialización de MediaPipe y variables del juego
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Inicialización de la ventana de OpenCV
cap = cv2.VideoCapture(0)
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Inicialización de la pelota
ball_pos = [random.randint(BALL_SIZE, cap_width - BALL_SIZE), random.randint(BALL_SIZE, cap_height - BALL_SIZE)]
ball_dir = [random.choice([-1, 1]) * BALL_SPEED, random.choice([-1, 1]) * BALL_SPEED]
ball_color = (0, 255, 0)

# Inicialización del juego
start_time = time.time()
score = 0
game_over = False
result_text = ""

def play_sound(frequency, duration=100):
    winsound.Beep(frequency, duration)

game_started = False
time_left = GAME_TIME
#Crear una ventana
cv2.namedWindow('Juego de Realidad virtualPinBall - By Nestor Balich')
cv2.setWindowProperty('Juego de Realidad virtualPinBall - By Nestor Balich', cv2.WND_PROP_TOPMOST, 1)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
        
    if game_started:    
        current_time = time.time()
        elapsed_time = current_time - start_time
        time_left = GAME_TIME - int(elapsed_time)
        
        if time_left <= 0 and not game_over:
            game_over = True
            game_started = False
            if score < TARGET_COUNT:
                result_text = "Perdiste"
            else:
                result_text = "Ganaste"

        if not game_over:
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    
                    index_pos = (int(index_finger_tip.x * cap_width), int(index_finger_tip.y * cap_height))
                    thumb_pos = (int(thumb_tip.x * cap_width), int(thumb_tip.y * cap_height))
                    
                    if np.linalg.norm(np.array(index_pos) - np.array(thumb_pos)) < BALL_SIZE:
                        if np.linalg.norm(np.array(index_pos) - np.array(ball_pos)) < BALL_SIZE:
                            score += 1
                            ball_color = (0, 0, 255)
                            ball_dir = [-ball_dir[0], -ball_dir[1]]
                            ball_dir[0] += SPEED_INCREMENT if ball_dir[0] > 0 else -SPEED_INCREMENT
                            ball_dir[1] += SPEED_INCREMENT if ball_dir[1] > 0 else -SPEED_INCREMENT
                            play_sound(700)  # Sonido de puntos
                        else:
                            ball_color = (0, 255, 0)

            ball_pos[0] += ball_dir[0]
            ball_pos[1] += ball_dir[1]

            if ball_pos[0] <= BALL_SIZE or ball_pos[0] >= cap_width - BALL_SIZE:
                ball_dir[0] *= -1
                play_sound(500)  # Sonido de rebote
            if ball_pos[1] <= BALL_SIZE or ball_pos[1] >= cap_height - BALL_SIZE:
                ball_dir[1] *= -1
                play_sound(500)  # Sonido de rebote

            cv2.circle(frame, tuple(ball_pos), BALL_SIZE, ball_color, -1)

    if game_over:
        if result_text=="Perdiste":
            result_color = (0, 0, 255)
        else:
            result_color = (0, 255, 0)
        
        cv2.putText(frame, result_text, (cap_width // 2 - 100, cap_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, result_color, 8)

    cv2.putText(frame, f"Tiempo: {max(time_left, 0)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Puntos: {score}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Juego de Realidad virtualPinBall - By Nestor Balich', frame)

    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # Tecla 'Esc' para salir
        break
    elif key == ord('s'):  # Tecla 'c' para borrar todo
        if game_started==False:
            game_started = True
            game_over = False
            score = 0
            time_left = GAME_TIME
            start_time = time.time()
            # Inicialización de la pelota
            ball_pos = [random.randint(BALL_SIZE, cap_width - BALL_SIZE), random.randint(BALL_SIZE, cap_height - BALL_SIZE)]
            ball_dir = [random.choice([-1, 1]) * BALL_SPEED, random.choice([-1, 1]) * BALL_SPEED]
            ball_color = (0, 255, 0)


cap.release()
cv2.destroyAllWindows()
