from djitellopy import Tello
import cv2, time, os, sys, signal, platform
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

OS = platform.system()
USAR_PYNPUT = (OS == "Darwin")  # macOS
print(f"USAR PYNPUT: {USAR_PYNPUT}")
if USAR_PYNPUT:
    from pynput import keyboard
    keys = set()

    def en_press(key):
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

    listener = keyboard.Listener(on_press=en_press, on_release=on_release)
    listener.start()


tello = Tello()
tello.connect()
print("Bateria:", tello.get_battery())

tello.streamoff()
tello.streamon()
frame_read = tello.get_frame_read()
time.sleep(2)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join("images", timestamp)
os.makedirs(save_dir, exist_ok=True)

tello.takeoff()
time.sleep(2)

def a_tierra():
    print("Aterrizando...")
    for _ in range(5):
        tello.send_rc_control(0,0,0,0)
        time.sleep(0.05)

    time.sleep(0.3)

    for _ in range(3):
        try:
            tello.land()
            print("Aterrizado OK")
            return
        except:
            time.sleep(0.5)

    print("Emergencia FORZADA")
    tello.emergency()

def salida(sig, imagen):
    a_tierra()
    tello.streamoff()
    tello.end()
    sys.exit(0)

signal.signal(signal.SIGINT, salida)

fps = 5
intervalo = 1.0 / fps
tiempo_de_imagen = time.time()
imagen_id = 0

velocidad = 40

# Control remoto (CR)
tiempo_cr = 0
cr_intervalo = 0.05

CONEXIONES_MANO = [
    (0,1),(1,2),(2,3),(3,4),      # Pulgar
    (0,5),(5,6),(6,7),(7,8),      # Índice
    (0,9),(9,10),(10,11),(11,12), # Medio
    (0,13),(13,14),(14,15),(15,16), # Anular
    (0,17),(17,18),(18,19),(19,20)  # Meñique
]

class Detector_de_Gestos:
    def __init__(self, ruta_modelo='gesture_recognizer.task'):
        opciones_basicas = python.BaseOptions(model_asset_path=ruta_modelo)
        opciones = vision.GestureRecognizerOptions(
            base_options=opciones_basicas,
            running_mode=vision.RunningMode.IMAGE
        )
        self.reconocedor = vision.GestureRecognizer.create_from_options(opciones)

    def dibujar_landmarks(self, frame, landmarks):
        h, w, _ = frame.shape
        puntos_px = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        for conexion in CONEXIONES_MANO:
            p1, p2 = puntos_px[conexion[0]], puntos_px[conexion[1]]
            cv2.line(frame, p1, p2, (255, 255, 255), 2)

        for pt in puntos_px:
            cv2.circle(frame, pt, 5, (255, 255, 0), -1)

    def obtener_direccion(self, landmarks):        
        idx_tip, idx_base = landmarks[8], landmarks[5]
        dx, dy = idx_tip.x - idx_base.x, idx_tip.y - idx_base.y
        if abs(dx) > abs(dy):
            if dx > 0.06: return "DERECHA"
            elif dx < -0.06: return "IZQUIERDA"
        else:
            if dy > 0.06: return "ABAJO"
            elif dy < -0.06: return "ARRIBA"
        return None
    
    def procesar_imagen(self, imagen):
        rgb_imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_imagen)
        resultado = self.reconocedor.recognize(mp_image)
        etiqueta = "Esperando..."

        if resultado.hand_landmarks:
            self.dibujar_landmarks(imagen, resultado.hand_landmarks[0])
            gesto = resultado.gestures[0][0].category_name
            if gesto == "Closed_Fist": etiqueta = "MANO CERRADA"
            elif gesto in ["Open_Palm", "Victory"]: etiqueta = "MANO ABIERTA"
            else:
                dir_label = self.obtener_direccion(resultado.hand_landmarks[0])
                etiqueta = dir_label if dir_label else "SEÑALANDO"

        return imagen, etiqueta
    
detector = Detector_de_Gestos()
    
if __name__ == "__main__":
    detector = Detector_de_Gestos()
    camara = cv2.VideoCapture(0)
    while camara.isOpened():
        ret, imagen = camara.read()
        if not ret: break
        imagen = cv2.flip(imagen, 1)
        imagen, etiqueta = detector.procesar_imagen(imagen)
        cv2.putText(imagen, etiqueta, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Deteccion Tasks UOH (Opcion A)', imagen)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

while True:
    imagen = frame_read.frame
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    if imagen is None: 
        continue

    imagen, etiqueta = detector.procesar_imagen(imagen)

    cv2.putText(imagen, f"GESTO: {etiqueta}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Tello", imagen)    
    
    if USAR_PYNPUT:
        cv2.pollKey()
        pressed = keys.copy()
    else:
        key = cv2.waitKey(1) & 0xFF
        pressed = set()
        if key != 255:
            pressed.add(chr(key))

    ahora = time.time()
    if ahora - tiempo_de_imagen >= intervalo:
        cv2.imwrite(os.path.join(save_dir, f"{imagen_id:06d}.png"), imagen)
        imagen_id += 1
        tiempo_de_imagen = ahora
    
    lr, fb, ud, yaw = 0, 0, 0, 0

    if etiqueta == "MANO ABIERTA":
        ud = velocidad
    elif etiqueta == "MANO CERRADA":
        ud = -velocidad
    elif etiqueta == "DERECHA":
        lr = velocidad
    elif etiqueta == "IZQUIERDA":
        lr = -velocidad
    elif etiqueta == "ARRIBA":
        fb = velocidad
    elif etiqueta == "ABAJO":
        fb = -velocidad

    if 'w' in pressed: fb = velocidad
    if 's' in pressed: fb = -velocidad
    if 'a' in pressed: lr = -velocidad
    if 'd' in pressed: lr = velocidad
    if 'r' in pressed: ud = velocidad
    if 'f' in pressed: ud = -velocidad
    if 'q' in pressed: yaw = -velocidad
    if 'e' in pressed: yaw = velocidad

    if ahora - tiempo_cr > cr_intervalo:
        tello.send_rc_control(lr, fb, ud, yaw)
        tiempo_cr = ahora

    if 'l' in pressed or 'esc' in pressed:
        a_tierra()
        break
    
tello.streamoff()
tello.end()    
cv2.destroyAllWindows()
camara.release()

