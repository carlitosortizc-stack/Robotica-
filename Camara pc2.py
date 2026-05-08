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
    running_mode=vision.RunningMode.VIDEO 
)

def obtener_etiqueta_gesto(resultado):
    """Interpreta el resultado para devolver uno de los 9 gestos totales"""
    if not resultado.gestures:
        return "Buscando mano..."

    # El nombre que viene directamente del modelo pre-entrenado
    nombre_gesto = resultado.gestures[0][0].category_name
    landmarks = resultado.hand_landmarks[0]

    # 1. Gestos estáticos nativos del modelo .task
    # "ILoveYou" es el nombre técnico que usa MediaPipe para el gesto de rock
    if nombre_gesto == "ILoveYou":
        return "ROCK / METAL"
    if nombre_gesto == "Thumb_Up":
        return "PULGAR ARRIBA (OK)"
    if nombre_gesto == "Thumb_Down":
        return "PULGAR ABAJO (STOP)"
    if nombre_gesto == "Closed_Fist":
        return "MANO CERRADA"
    if nombre_gesto in ["Open_Palm", "Victory"]:
        return "MANO ABIERTA"

    # 2. Lógica de Direcciones (Calculada por nosotros)
    # Solo llegamos aquí si no es ninguno de los gestos anteriores
    idx_tip, idx_base = landmarks[8], landmarks[5]
    dx, dy = idx_tip.x - idx_base.x, idx_tip.y - idx_base.y
    umbral = 0.05
    
    if abs(dx) > abs(dy):
        return "DERECHA" if dx > umbral else "IZQUIERDA" if dx < -umbral else "SEÑALANDO"
    else:
        return "ABAJO" if dy > umbral else "ARRIBA" if dy < -umbral else "SEÑALANDO"

# --- BUCLE PRINCIPAL ---
cap = cv2.VideoCapture(0)

with vision.GestureRecognizer.create_from_options(options) as recognizer:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Espejo y conversión a RGB (MediaPipe moderno requiere RGB)
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # En modo VIDEO, necesitamos un timestamp en milisegundos
        ms_timestamp = int(time.time() * 1000)
        
        # Realizar la detección
        resultado_deteccion = recognizer.recognize_for_video(mp_image, ms_timestamp)

        # Obtener la etiqueta de los gestos
        texto_gesto = obtener_etiqueta_gesto(resultado_deteccion)

        # --- MOSTRAR RESULTADO ---
        # Rectángulo negro para fondo del texto
        cv2.rectangle(frame, (0, 0), (450, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"COMANDO: {texto_gesto}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Robotica UOH - Gesto Rock Añadido', frame)

        # Salir con la tecla ESC (27) o 'q'
        if cv2.waitKey(5) & 0xFF in [27, ord('q')]:
            break

cap.release()
cv2.destroyAllWindows()