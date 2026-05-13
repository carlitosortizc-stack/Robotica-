import cv2
import mediapipe as mp
import numpy as np
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pupil_apriltags import Detector

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
            running_mode=vision.RunningMode.VIDEO # Cambiado a VIDEO para mayor fluidez
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

    def draw_manual_landmarks(self, frame, landmarks):
        h, w, _ = frame.shape
        puntos_px = []
        for lm in landmarks:
            puntos_px.append((int(lm.x * w), int(lm.y * h)))
        for conexion in CONEXIONES_MANO:
            p1 = puntos_px[conexion[0]]
            p2 = puntos_px[conexion[1]]
            cv2.line(frame, p1, p2, (255, 255, 255), 2)
        for pt in puntos_px:
            cv2.circle(frame, pt, 5, (255, 255, 0), -1)

    def get_direction(self, landmarks):
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
        # MediaPipe necesita RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        # Usamos marca de tiempo para el modo VIDEO
        result = self.recognizer.recognize_for_video(mp_image, int(time.time() * 1000))
        label = "Esperando..."
        if result.hand_landmarks:
            self.draw_manual_landmarks(frame, result.hand_landmarks[0])
            gesture_name = result.gestures[0][0].category_name
            if gesture_name == "Thumb_Up": label = "PULGAR ARRIBA"
            elif gesture_name == "Thumb_Down": label = "PULGAR ABAJO"
            elif gesture_name == "ILoveYou": label = "ROCK / METAL"
            elif gesture_name == "Closed_Fist": label = "MANO CERRADA"
            elif gesture_name in ["Open_Palm", "Victory"]: label = "MANO ABIERTA"
            else:
                dir_label = self.get_direction(result.hand_landmarks[0])
                label = dir_label if dir_label else "SEÑALANDO"
        return frame, label

# --- INICIALIZACIÓN DE DETECTORES ---
detector_tags = Detector(families="tag36h11")
ancho_cam, alto_cam = 640, 480 # Resoluciones estándar suelen funcionar mejor
centro_x, centro_y = ancho_cam // 2, alto_cam // 2
kp_tags = 0.1

if __name__ == "__main__":
    detector = GestureDetector()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho_cam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto_cam)
    
    print("Iniciando Robótica UOH... Presiona 'q' para salir.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 1. El detector de Tags necesita la imagen en escala de grises y ORIGINAL (sin flip)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resultado_tags = detector_tags.detect(gray)
        
        # 2. Ahora aplicamos el efecto espejo para que tú te veas bien
        frame = cv2.flip(frame, 1)
        
        # 3. Procesamos gestos
        frame, etiqueta = detector.process_frame(frame)
        
        apriltags_detectado = False

        # 4. Lógica de Tags CORREGIDA (es una lista, no tiene .detections)
        if len(resultado_tags) > 0:
            tag = resultado_tags[0] # Seguimos el primer tag
            
            # CORRECCIÓN DE COORDENADAS POR EL FLIP:
            # Si el frame está invertido, la X del tag también debe invertirse
            cx = ancho_cam - int(tag.center[0])
            cy = int(tag.center[1])

            apriltags_detectado = True

            # Dibujar en pantalla
            cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 25, 2)
            cv2.putText(frame, f"ID: {tag.tag_id}", (cx + 15, cy - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # --- MENSAJE DE SEGUIMIENTO ---
            # Dibujamos un cuadro verde con el texto SIGUIENDO
            cv2.rectangle(frame, (centro_x - 100, alto_cam - 70), (centro_x + 100, alto_cam - 20), (0, 120, 0), -1)
            cv2.putText(frame, "SIGUIENDO", (centro_x - 85, alto_cam - 35), 
                        cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2)

        # UI: Cuadro de información de gestos
        cv2.rectangle(frame, (0, 0), (350, 60), (0, 0, 0), -1)
        cv2.putText(frame, etiqueta, (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if not apriltags_detectado:
            cv2.putText(frame, "BUSCANDO TAG...", (ancho_cam - 200, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow('Robotica UOH - Sistema Integrado', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()