import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURACIÓN DEL RECONOCEDOR (API MODERNA) ---
model_path = 'gesture_recognizer.task' 

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO # Optimizado para webcam
)

def obtener_etiqueta_gesto(resultado):
    """Interpreta el resultado para devolver uno de los 6 gestos"""
    if not resultado.gestures:
        return "Buscando mano..."

    nombre_gesto = resultado.gestures[0][0].category_name
    landmarks = resultado.hand_landmarks[0]

    # 1. Gestos de Mano Abierta y Cerrada
    if nombre_gesto == "Closed_Fist":
        return "MANO CERRADA"
    if nombre_gesto in ["Open_Palm", "Victory"]:
        return "MANO ABIERTA"

    # 2. Lógica de Direcciones (Arriba, Abajo, Izquierda, Derecha)
    # Comparamos punta del índice (8) con su base (5)
    idx_tip, idx_base = landmarks[8], landmarks[5]
    dx, dy = idx_tip.x - idx_base.x, idx_tip.y - idx_base.y
    umbral = 0.05
    
    if abs(dx) > abs(dy):
        return "DERECHA" if dx > umbral else "IZQUIERDA" if dx < -umbral else "SEÑALANDO"
    else:
        return "ABAJO" if dy > umbral else "ARRIBA" if dy < -umbral else "SEÑALANDO"

# --- BUCLE PRINCIPAL (TU ESTRUCTURA ADAPTADA) ---
cap = cv2.VideoCapture(0)

# Inicializamos el reconocedor usando un 'with' para que se cierre solo al terminar
with vision.GestureRecognizer.create_from_options(options) as recognizer:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignorando frame vacío de la cámara.")
            continue

        # Espejo y conversión a RGB (MediaPipe moderno requiere RGB)
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # En modo VIDEO, necesitamos un timestamp en milisegundos
        ms_timestamp = int(time.time() * 1000)
        
        # Realizar la detección
        resultado_deteccion = recognizer.recognize_for_video(mp_image, ms_timestamp)

        # Obtener la etiqueta de los 6 gestos
        texto_gesto = obtener_etiqueta_gesto(resultado_deteccion)

        # --- MOSTRAR RESULTADO ---
        # Dibujamos un rectángulo negro para que el texto se lea bien
        cv2.rectangle(frame, (0, 0), (380, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"GESTO: {texto_gesto}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Detección Tasks UOH (Opción A)', frame)

        # Salir con la tecla ESC (27) o 'q'
        if cv2.waitKey(5) & 0xFF in [27, ord('q')]:
            break

cap.release()
cv2.destroyAllWindows()