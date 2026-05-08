import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURACIÓN DE ESTRUCTURA ---
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

        for lm in landmarks:
            puntos_px.append((int(lm.x * w), int(lm.y * h)))

        # Dibujar líneas (Blanco)
        for conexion in CONEXIONES_MANO:
            p1 = puntos_px[conexion[0]]
            p2 = puntos_px[conexion[1]]
            cv2.line(frame, p1, p2, (255, 255, 255), 2)

        # Dibujar puntos (Cian)
        for pt in puntos_px:
            cv2.circle(frame, pt, 5, (255, 255, 0), -1)

    def get_direction(self, landmarks):
        """Lógica para Arriba, Abajo, Izquierda, Derecha"""
        idx_tip, idx_base = landmarks[8], landmarks[5]
        dx, dy = idx_tip.x - idx_base.x, idx_tip.y - idx_base.y
        umbral = 0.06
        
        if abs(dx) > abs(dy):
            if dx > umbral: return "DERECHA"
            elif dx < -umbral: return "IZQUIERDA"
        else:
            if dy > umbral: return "ABAJO"
            elif dy < -umbral: return "ARRIBA"
        return None

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.recognizer.recognize(mp_image)
        label = "Esperando..."

        if result.hand_landmarks:
            # 1. Dibujar la estructura visual
            self.draw_manual_landmarks(frame, result.hand_landmarks[0])

            # 2. Lógica de Gestos (Prioridad a los 9 gestos totales)
            gesture_name = result.gestures[0][0].category_name
            
            if gesture_name == "Thumb_Up":
                label = "PULGAR ARRIBA"
            elif gesture_name == "Thumb_Down":
                label = "PULGAR ABAJO"
            elif gesture_name == "ILoveYou":
                label = "ROCK / METAL"
            elif gesture_name == "Closed_Fist":
                label = "MANO CERRADA"
            elif gesture_name in ["Open_Palm", "Victory"]:
                label = "MANO ABIERTA"
            else:
                # Si no es ninguno de los anteriores, calcular dirección
                dir_label = self.get_direction(result.hand_landmarks[0])
                label = dir_label if dir_label else "SEÑALANDO"

        return frame, label

# --- EJECUCIÓN ---
if __name__ == "__main__":
    detector = GestureDetector()
    cap = cv2.VideoCapture(0)
    
    print("Iniciando... Presiona 'q' para salir.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        frame, etiqueta = detector.process_frame(frame)
        
        # UI: Cuadro de información
        cv2.rectangle(frame, (0, 0), (400, 60), (0, 0, 0), -1)
        cv2.putText(frame, etiqueta, (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Robotica UOH - 9 Gestos Manual', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
