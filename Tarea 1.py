from djitellopy import Tello
import cv2, time, os, sys, signal, platform
from datetime import datetime


import mediapipe as mp

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
        base_options = python.BaseOptions(
                        model_asset_path=model_path, 
                        # delegate=python.BaseOptions.Delegate.GPU
                                          )

        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO
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
        # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = cv2.resize(frame, (int(frame.shape[0]), int(frame.shape[1])))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # result = self.recognizer.recognize(mp_image)
        result = self.recognizer.recognize_for_video(mp_image, int(time.time()*1000))
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
            elif gesture_name == "Open_Palm":
                label = "MANO ABIERTA"
            elif gesture_name == "Victory":
                label = "PAZ"
            else:
                # Si no es ninguno de los anteriores, calcular dirección
                dir_label = self.get_direction(result.hand_landmarks[0])
                label = dir_label if dir_label else "SEÑALANDO"

        return frame, label

# =======================
# DETECTAR OS
# =======================
OS = platform.system()
USE_PYNPUT = (OS == "Darwin")  # macOS
print(f"USE PYNPUT: {USE_PYNPUT}")
if USE_PYNPUT:
    from pynput import keyboard
    keys = set()

    def on_press(key):
        try:
            keys.add(key.char)
        except:
            if key == keyboard.Key.esc:
                keys.add('esc')

    def on_release(key):
        try:
            keys.discard(key.char)
        except:
            if key == keyboard.Key.esc:
                keys.discard('esc')

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

# =======================
# TELLO
# =======================
tello = Tello()
tello.connect()
print("Battery:", tello.get_battery())

print(f"\n\n ENCENDIDO \n\n")

tello.streamoff()
tello.streamon()
frame_read = tello.get_frame_read()
time.sleep(2)

# =======================
# SAVE DIR
# =======================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join("images", timestamp)
os.makedirs(save_dir, exist_ok=True)

# =======================
# TAKEOFF
# =======================
#tello.takeoff()
#time.sleep(2)

# =======================
# SAFE LAND FUNCTION
# =======================
def safe_land():
    print("LANDING...")

    # detener rc
    for _ in range(5):
        tello.send_rc_control(0,0,0,0)
        time.sleep(0.05)

    time.sleep(0.3)

    # intentar land varias veces
    for _ in range(3):
        try:
            tello.land()
            print("LANDED OK")
            return
        except:
            time.sleep(0.5)

    print("FORCED EMERGENCY")
    tello.emergency()

# =======================
# CTRL+C
# =======================
def handler(sig, frame):
    safe_land()
    tello.streamoff()
    tello.end()
    sys.exit(0)

signal.signal(signal.SIGINT, handler)

# =======================
# LOOP
# =======================
fps = 5
interval = 1.0 / fps
last_frame_time = time.time()
frame_id = 0

speed = 40

# RC rate limit
last_rc_time = 0
rc_interval = 0.05

detector = GestureDetector()


while True:
    frame = frame_read.frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if frame is None:
        continue
    frame, etiqueta = detector.process_frame(frame)
        # UI: Cuadro de información
    cv2.rectangle(frame, (0, 0), (400, 60), (0, 0, 0), -1)
    cv2.putText(frame, etiqueta, (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # =======================
    # DISPLAY
    # =======================
    cv2.imshow("Tello", frame)

    if USE_PYNPUT:
        cv2.pollKey()
        pressed = keys.copy()
    else:
        key = cv2.waitKey(1) & 0xFF
        pressed = set()
        if key != 255:
            pressed.add(chr(key))

    # =======================
    # SAVE
    # =======================
    now = time.time()
    if now - last_frame_time >= interval:
        cv2.imwrite(os.path.join(save_dir, f"{frame_id:06d}.png"), frame)
        frame_id += 1
        last_frame_time = now

    # =======================
    # CONTROL
    # =======================
    lr, fb, ud, yaw = 0, 0, 0, 0



    if etiqueta == "ARRIBA":
        ud=20
    elif etiqueta == "ABAJO":
        ud=-20
    elif etiqueta == "DERECHA":
        lr=20
    elif etiqueta == "IZQUIERDA":
        lr=-20
    elif etiqueta == "PULGAR ARRIBA":
        tello.flip_back()
        lr, fb, ud, yaw = 0, 0, 0, 0
        time.sleep(2)
    elif etiqueta == "PULGAR ABAJO":
        tello.flip_forward()
        lr, fb, ud, yaw = 0, 0, 0, 0
        time.sleep(2)
    elif etiqueta == "MANO ABIERTA":
        tello.takeoff()
        lr, fb, ud, yaw = 0, 0, 0, 0
        time.sleep(1)
    elif etiqueta == "MANO CERRADA":
        tello.land()
        lr, fb, ud, yaw = 0, 0, 0, 0
        time.sleep(1)
    
   #elif etiqueta == "PAZ":
   #    tello.emergency()
   #    lr, fb, ud, yaw = 0, 0, 0, 0
   #    time.sleep(2)


    if 'w' in pressed: fb = speed
    if 's' in pressed: fb = -speed
    if 'a' in pressed: lr = -speed
    if 'd' in pressed: lr = speed
    if 'r' in pressed: ud = speed
    if 'f' in pressed: ud = -speed
    if 'q' in pressed: yaw = -speed
    if 'e' in pressed: yaw = speed

    print(f"{lr=}, {fb=}, {ud=}, {yaw=}")

    # =======================
    # SEND RC
    # =======================
    if now - last_rc_time > rc_interval:
        tello.send_rc_control(lr, fb, ud, yaw)
        last_rc_time = now

    # =======================
    # LAND
    # =======================
    if 'l' in pressed or 'esc' in pressed:
        safe_land()
        break



# =======================
# CLEANUP
# =======================
tello.end()
cv2.destroyAllWindows()
