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

def interpretar_gesto(resultado):
    if not resultado.gestures:
        return "Buscando mano..."
    nombre_gesto = resultado.gestures[0][0].category_name
    landmarks = resultado.hand_landmarks[0]
    if nombre_gesto == "Closed_Fist":
        return "MANO CERRADA"
    if nombre_gesto in ["Open_Palm", "Victory"]:
        return "MANO ABIERTA"
    idx_tip, idx_base = landmarks[8], landmarks[5]
    dx, dy = idx_tip.x - idx_base.x, idx_tip.y - idx_base.y
    umbral = 0.05
    
    if abs(dx) > abs(dy):
        return "DERECHA" if dx > umbral else "IZQUIERDA" if dx < -umbral else "SEÑALANDO"
    else:
        return "ABAJO" if dy > umbral else "ARRIBA" if dy < -umbral else "SEÑALANDO"

ruta_modelo = "gesture_recognizer.task"
opciones_basicas = python.BaseOptions(model_asset_path=ruta_modelo)
opciones = vision.GestureRecognizerOptions(
    base_options=opciones_basicas,
    running_mode=vision.RunningMode.VIDEO
)

reconocedor = vision.GestureRecognizer.create_from_options(opciones)

while True:
    imagen = frame_read.frame
    if imagen is None:
        continue

    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    camara =cv2.VideoCapture(0)
    with vision.GestureRecognizer.create_from_options(opciones) as reconocedor:
        while camara.isOpened():
            exito, frame = camara.read()
            if not exito:
                print("Ignorando imagen vacío de la cámara.")
                continue

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    ms_timestamp = int(time.time() * 1000)

    resultado_deteccion = reconocedor.recognize_for_video(mp_image, ms_timestamp)
    texto_gesto = interpretar_gesto(resultado_deteccion)

    cv2.rectangle(frame, (0, 0), (380, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"GESTO: {texto_gesto}", (20, 40), 
                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if cv2.waitKey(5) & 0xFF in [27, ord('q')]:
         break

    cv2.imshow("Tello", imagen)
    
    if USAR_PYNPUT:
        cv2.pollKey()
        pressed = keys.copy()
    else:
        key = cv2.waitKey(1) & 0xFF
        pressed = set()
        if key != 255:
            pressed.add(chr(key))
            if key == 27: pressed.add('esc')

    ahora = time.time()
    if ahora - tiempo_de_imagen >= intervalo:
        cv2.imwrite(os.path.join(save_dir, f"{imagen_id:06d}.png"), imagen)
        imagen_id += 1
        tiempo_de_imagen = ahora
    
    lr, fb, ud, yaw = 0, 0, 0, 0

    if texto_gesto == "MANO ABIERTA":
        fb = velocidad
    elif texto_gesto == "MANO CERRADA":
        fb = -velocidad
    elif texto_gesto == "DERECHA":
        lr = velocidad
    elif texto_gesto == "IZQUIERDA":
        lr = -velocidad
    elif texto_gesto == "ARRIBA":
        fb = velocidad
    elif texto_gesto == "ABAJO":
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

