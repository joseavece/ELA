import math as mt
import cv2
import mediapipe as mp
import time
import numpy as np
#0 para IRIUN, 1 para built-in
CAMARA = 0
CYAN = (255, 200, 0)
ORANGE = (39, 149, 245)

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

webcam = cv2.VideoCapture(CAMARA)

LEFT_EYE = [33, 133, 159, 145]
RIGHT_EYE = [263, 362, 386, 374]
LEFT_GAZE = [469, 470, 471, 472, 468]
RIGHT_GAZE = [474, 475, 476, 477, 473]

CONJUNTOS = [LEFT_EYE, RIGHT_EYE, LEFT_GAZE, RIGHT_GAZE]

latest_result = None

def distanciaPorcentual(origen, destino, actual):
  extremoAExtremo = distancia(origen, destino)
  origenAActual = distancia(origen, actual)
  actualADestino = distancia(destino, actual)

  origenAActualPorcentual = origenAActual/extremoAExtremo
  actualADestinoPorcentual = actualADestino/extremoAExtremo

  return origenAActualPorcentual, actualADestinoPorcentual



def distancia(punto1, punto2):
  x1 = punto1[0]
  y1 = punto1[1]
  x2 = punto2[0]
  y2 = punto2[1]
  distx = x1-x2
  disty = y1-y2
  mod = mt.sqrt(distx**2 + disty**2)
  return mod

def print_result(result: mp.tasks.vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    """Callback que recibe los resultados inferidos por MediaPipe en el modo LIVE_STREAM."""
    global latest_result
    latest_result = result

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_faces=1,
    result_callback=print_result
)

with FaceLandmarker.create_from_options(options) as landmarker:
    with open("datos.txt", "a") as f:
        while(webcam.isOpened()):
            ret, frame = webcam.read()

            if not ret:
                break

            #Flip de laimagen
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            #Conversión de la imagen de bgr a rgb y a Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            frame_timestamp_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, frame_timestamp_ms)

            if latest_result is not None and latest_result.face_landmarks:
                for face_landmarks in latest_result.face_landmarks:

                    #Dibujar landmarks en el ojo 
                    for eye in CONJUNTOS:
                        color = CYAN if eye in [LEFT_EYE, RIGHT_EYE] else ORANGE
                        for id in eye:
                            lm = face_landmarks[id]
                            cx, cy = int(lm.x * w), int(lm.y*h)
                            cv2.circle(frame, (cx, cy), 2, color, -1)

                            
                        if pressed_key==81 or pressed_key==113: #Si se presiona g ó G
                            datos = [(id, face_landmarks[id].x, face_landmarks[id].y) for id in RIGHT_EYE]
                            f.write("EAR izquierdo"+str(datos)+"\n")
                        if pressed_key==69 or pressed_key==101: #Si se presiona e ó E
                            datos = [(id, face_landmarks[id].x, face_landmarks[id].y) for id in LEFT_EYE]
                            f.write("EAR derecho"+str(datos)+"\n")
                        if pressed_key==65 or pressed_key==97: #Si se presiona a ó A
                            datos = [(id, face_landmarks[id].x, face_landmarks[id].y) for id in RIGHT_EYE]
                            f.write("Gaze izquierdo"+str(datos)+"\n")
                        if pressed_key==68 or pressed_key==100: #Si se presiona d ó D
                            datos = [(id, face_landmarks[id].x, face_landmarks[id].y) for id in RIGHT_EYE]
                            f.write("Gaze derecho"+str(datos)+"\n")
                        

            cv2.imshow("Prueba face", frame)

            pressed_key = cv2.waitKey(1)

            if pressed_key==27:
                break


webcam.release()
cv2.destroyAllWindows()
