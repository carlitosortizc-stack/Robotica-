import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Definimos las conexiones de la mano manualmente (dedo por dedo)
CONEXIONES_MANO = [
    (0, 1), (1, 2), (2, 3), (3, 4),    # Pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),    # Índice
    (5, 9), (9, 10), (10, 11), (11, 12), # Medio
    (9, 13), (13, 14), (14, 15), (15, 16), # Anular
    (13, 17), (17, 18), (18, 19), (19, 20), # Meñique
    (0, 17) # Palma
]

class GestureDetector:
    def __init__(self, model_path='gesture_recognizer.task'):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

    def draw_manual_landmarks(self, frame, landmarks):
        """Dibuja los puntos y líneas usando solo OpenCV"""
        h, w, _ = frame.shape
        puntos_px = []

        # 1. Convertir coordenadas normalizadas a píxeles
        for lm in landmarks:
            puntos_px.append((int(lm.x * w), int(lm.y * h)))

        # 2. Dibujar las líneas (Huesos) en Blanco
        for conexion in CONEXIONES_MANO:
            p1 = puntos_px[conexion[0]]
            p2 = puntos_px[conexion[1]]
            cv2.line(frame, p1, p2, (255, 255, 255), 2)

        # 3. Dibujar los puntos (Articulaciones) en Cian
        for pt in puntos_px:
            cv2.circle(frame, pt, 5, (255, 255, 0), -1)

    def get_direction(self, landmarks):
        idx_tip, idx_base = landmarks[8], landmarks[5]
        dx, dy = idx_tip.x - idx_base.x, idx_tip.y - idx_base.y
        if abs(dx) > abs(dy):
            if dx > 0.06: return "DERECHA"
            elif dx < -0.06: return "IZQUIERDA"
        else:
            if dy > 0.06: return "ABAJO"
            elif dy < -0.06: return "ARRIBA"
        return None

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.recognizer.recognize(mp_image)
        label = "Esperando..."

        if result.hand_landmarks:
            # DIBUJO MANUAL SIN MP.SOLUTIONS
            self.draw_manual_landmarks(frame, result.hand_landmarks[0])

            # GESTOS
            gesture_name = result.gestures[0][0].category_name
            if gesture_name == "Closed_Fist": label = "MANO CERRADA"
            elif gesture_name in ["Open_Palm", "Victory"]: label = "MANO ABIERTA"
            else:
                dir_label = self.get_direction(result.hand_landmarks[0])
                label = dir_label if dir_label else "SEÑALANDO"

        return frame, label

if __name__ == "__main__":
    detector = GestureDetector()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        frame, etiqueta = detector.process_frame(frame)
        cv2.putText(frame, etiqueta, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Robotica UOH - Fix Manual', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()