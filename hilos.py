import threading
import time
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from sort import Sort
import yolov5
import pytesseract
import easyocr
import os
import re
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from yolov5.models. common import DetectMultiBackend

# Variables globales para compartir datos entre hilos
frame_queue = []
results_vehicle_queue = []
results_plate_queue = []
lock = threading.Lock()


# Adquisición de video y reconocimiento de placas
def Hilo1():
    
    # Dimensiones de la placa en milímetros
    PLATE_WIDTH_MM = 303.6
    PLATE_WIDTH_MM -= 80
    PLATE_HEIGHT_MM = 151.2

    pixelametro = (20/1000)/137 #metros

    #PLATE_HEIGHT_MM -= 20
    initial_time = datetime.now()

    # Definir múltiples rangos de colores para el amarillo y el blanco en el espacio HSV
    color_ranges = [
        (np.array([20, 100, 100]), np.array([30, 255, 255])),  # Amarillo Rango 1
        (np.array([15, 100, 100]), np.array([25, 255, 255])),  # Amarillo Rango 2
        (np.array([25, 100, 100]), np.array([35, 255, 255])),  # Amarillo Rango 3
        (np.array([20, 100, 50]), np.array([30, 255, 200])),
        (np.array([15, 100, 50]), np.array([25, 255, 180])),
        (np.array([20, 150, 40]), np.array([30, 255, 150])),
        (np.array([0, 0, 200]), np.array([180, 25, 255])),      # Blanco
        (np.array([0, 0, 220]), np.array([180, 30, 255])),
        (np.array([0, 0, 180]), np.array([180, 40, 255])),
        (np.array([0, 0, 160]), np.array([180, 50, 255])),

        (np.array([0, 0, 0]), np.array([180, 255, 50])),
        (np.array([0, 0, 0]), np.array([180, 255, 70])),
        (np.array([0, 0, 0]), np.array([180, 50, 100]))
    ]


    plate_speeds = {}

    DISTANCE_VECTOR = np.linspace(45, 75, num=2160)  # Suponiendo un ancho de imagen de 2160 píxeles

    # Diccionario para almacenar el frame inicial de detección de la placa para cada track_id
    plate_initial_frames = {}


    # Función para calcular la velocidad basada en el tamaño de la placa
    def calPlateSpeed(plate_width_pixels, video_fps, frame_count,relacion,plate_y_position):
        # Supongamos una distancia promedio entre la cámara y los vehículos de 60
        DISTANCE_TO_CAMERA_M = 60
        DISTANCE_TO_CAMERA_M = DISTANCE_VECTOR[plate_y_position]

        # Calcular el tamaño de la placa en metros
        plate_width_meters = (PLATE_WIDTH_MM / 1000) / DISTANCE_TO_CAMERA_M * plate_width_pixels

        print(plate_width_meters)

        #Velocidad = Distancia / Tiempo
        # Calcular la velocidad como un cambio en el tamaño de la placa entre frames consecutivos
        speed = 600 / video_fps * plate_width_meters * 3.6   # Convertir a km/h
        print(f"Speed: {speed}")
        return round(abs(speed), 2)


    if __name__ == "__main__":
        #cap = cv2.VideoCapture("/home/javier/ProyectoFinal/videosfinales/Videos/MVI_7857.MP4")
        #initial_frame_number = 600 # Cambia este valor al frame deseado 2850

        cap = cv2.VideoCapture("/home/javier/ProyectoFinal/videosfinales/Videos/MVI_7861.MP4")
        initial_frame_number = 570

        #cap = cv2.VideoCapture("/home/javier/ProyectoFinal/videosfinales/Videos/MVI_7862.MP4")
        #initial_frame_number = 529

        VIDEO_FPS = cap.get(cv2.CAP_PROP_FPS)
        print(f"Procesando video a {VIDEO_FPS} FPS")

        # Establecer el frame inicial
        cap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame_number)
        
        # Definir el frame final deseado
        final_frame_number = 1000

        # Modelos YOLO
        model_vehicle = YOLO("yolov8n.pt")
        model_plate = YOLO("license_plate_detector.pt")

        tracker = Sort()
        prev_frame = None
        prev_boxes = {}

        # Leer el primer frame
        ret, initial_frame = cap.read()

        prev_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
        initial_frame = prev_frame


        while cap.isOpened():
            status, frame = cap.read()

            if not status:
                break

            if frame is None or frame.size == 0:
                print("Frame vacío, saltando...")
                continue
            
            
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Verificar si se ha alcanzado el frame final
            if current_frame >= final_frame_number:
                break

            # Convertir el frame actual a escala de grises
            #curr_gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Convertir el frame actual a espacio de color HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Aplicar las máscaras para resaltar los tonos de amarillo y blanco
            mask = np.zeros(hsv.shape[:2], dtype="uint8")
            for lower, upper in color_ranges:
                mask += cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            results_vehicle = model_vehicle(frame, stream=True)
            results_plate = model_plate(frame, stream=True)

            # Detección y seguimiento de vehículos
            for res in results_vehicle:
                filtered_indices = np.where((np.isin(res.boxes.cls.cpu().numpy(), [2, 5, 7])) & (res.boxes.conf.cpu().numpy() > 0.3))[0]
                boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)

                tracks = tracker.update(boxes)
                tracks = tracks.astype(int)

                for xmin, ymin, xmax, ymax, track_id in tracks:
                    xc, yc = int((xmin + xmax) / 2), ymax

                    prev_boxes[track_id] = [xmin, ymin, xmax, ymax]

                    # Detección de la placa
                for plate_res in results_plate:
                    plate_boxes = plate_res.boxes.xyxy.cpu().numpy().astype(int)
                    for plate_box in plate_boxes:
                        p_xmin, p_ymin, p_xmax, p_ymax = plate_box
                        plate_width_pixels = p_xmax - p_xmin
                        #plate_width_pixels *= 1.15 
                        plate_heigth_pixels = p_ymax - p_ymin
                        plate_x_position = (p_ymin + p_ymax) // 2  # Calcular la posición Y de la placa

                        # Almacenar el frame inicial de detección de la placa
                        if track_id not in plate_initial_frames:
                            plate_initial_frames[track_id] = current_frame

                        # Calcular la diferencia de frames
                        frame_count = current_frame - plate_initial_frames[track_id]

                        relacion = (xmax - xmin) / (p_xmax - p_xmin)
                        print(relacion)

                        # Calcular velocidad basada en la placa
                        plate_speed = calPlateSpeed(plate_width_pixels, VIDEO_FPS, frame_count,relacion, plate_x_position)
                        plate_speeds[track_id] = f"{plate_speed} Km/h"

                        # Dibujar el recuadro de la placa
                        cv2.rectangle(frame, (p_xmin, p_ymin), (p_xmax, p_ymax), (255, 0, 255), 2)
                        cv2.putText(frame, plate_speeds[track_id], (p_xmin, p_ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.9, (0, 0, 255), 2)

                    # Mostrar las velocidades calculadas
                    if track_id in plate_speeds:
                        cv2.putText(frame, plate_speeds[track_id], (xmin, ymin - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

                    cv2.circle(frame, (xc, yc), 5, (0, 255, 0), -1)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)

            frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
            # Mostrar el frame
            cv2.imshow("Result", frame)
            result = cv2.resize(result, (0, 0), fx=0.3, fy=0.3)
            #cv2.imshow("Yellow", result)

            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

# Tracking
def Hilo2():
    # Cargar el modelo preentrenado de YOLOv5 para la detección de matrículas
    model = yolov5.load('keremberke/yolov5m-license-plate')

    # Especificar la carpeta que contiene los videos
    videos_folder = '/home/javier/ProyectoFinal/videosfinales/Videos'

    # Obtener la lista de archivos de video en la carpeta
    video_files = [f for f in os.listdir(videos_folder) if f.endswith('.MP4')]
    if not video_files:
        print("No se encontró ningún archivo de video en la carpeta.")
        exit()


    video_index = 3;
    video_file = video_files[video_index]
    video_path = os.path.join(videos_folder, video_file)

    # Configuración personalizada para Tesseract
    custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    # Expresión regular para validar el patrón LLLNNN
    pattern = re.compile(r'^[A-Z]{3}[0-9]{3}$')

    # Inicializar el lector de EasyOCR
    reader = easyocr.Reader(['en'])

    # Cargar el video
    cap = cv2.VideoCapture(video_path)

    # Verificar si el video se abrió correctamente
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el archivo de video {video_file}")
        exit()

    # Obtener las dimensiones del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Definir la ROI (por ejemplo, la mitad inferior del video)
    roi_top = 0
    roi_bottom = height // 2
    roi_left = 0
    roi_right = width

    # Establecer la tasa de fotogramas a 24 fps
    fps = 24
    frame_interval = 1.0 / fps  # Intervalo de tiempo entre fotogramas en segundos
    print(f"Procesando {video_file} a {fps} FPS y resolución {width}x{height}")

    frame_number = 0

    # Procesar cada fotograma del video
    while cap.isOpened():
        start_time = time.time()  # Medir tiempo de inicio del frame
        
        ret, frame = cap.read()
        if not ret:
            break  # Salir del bucle si no hay más fotogramas disponibles

        frame_number += 1

        # Aplicar la ROI al fotograma
        roi_frame = frame[roi_top:roi_bottom, roi_left:roi_right]

        # Realizar la detección en el fotograma con ROI
        results = model(roi_frame)

        # Procesar cada matrícula detectada en el fotograma
        for result in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Ajustar las coordenadas de la detección a la ROI
            x1 += roi_left
            y1 += roi_top
            x2 += roi_left
            y2 += roi_top
            
            # Recortar la región de la matrícula
            plate_region = frame[y1:y2, x1:x2]
            
            # Ajustar la ROI para enfocarse más en el área central de la placa
            margin_x = int((x2 - x1) * 0.1)
            margin_y = int((y2 - y1) * 0.1)
            x1 = max(x1 + margin_x, 0)
            y1 = max(y1 + margin_y, 0)
            x2 = min(x2 - margin_x, frame.shape[1])
            y2 = min(y2 - margin_y, frame.shape[0])
            plate_region = frame[y1:y2, x1:x2]
            
            plate_region = cv2.resize(plate_region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # Aumentar el brillo
            brightness = 50
            bright_image = cv2.convertScaleAbs(plate_region, alpha=1, beta=brightness)
            
            # Aumentar el contraste
            contrast = 1.5
            contrast_image = cv2.convertScaleAbs(bright_image, alpha=contrast, beta=0)
            
            # Aumentar la nitidez
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharp_image = cv2.filter2D(contrast_image, -1, kernel)
            
            # Aplicar OCR con Tesseract
            text_tesseract = pytesseract.image_to_string(sharp_image, config=custom_config).strip()
            
            # Aplicar OCR con EasyOCR
            result_easyocr = reader.readtext(sharp_image, detail=0)
            text_easyocr = result_easyocr[0] if result_easyocr else ""
            
            # Filtrar el texto de EasyOCR para que solo contenga letras mayúsculas y números
            text_easyocr = ''.join(filter(lambda x: x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', text_easyocr))
            
            # Validar el texto reconocido con el patrón LLLNNN y longitud de 6 caracteres
            if pattern.match(text_tesseract) and len(text_tesseract) == 6:
                print(f'Tesseract detectó: {text_tesseract}')
            if pattern.match(text_easyocr) and len(text_easyocr) == 6:
                print(f'EasyOCR detectó: {text_easyocr}')
            
            # Verificar si la placa contiene caracteres 'Q' o 'O'
            contains_similar_chars = 'Q' in text_tesseract or 'O' in text_tesseract or 'Q' in text_easyocr or 'O' in text_easyocr
            
            # Determinar el color del bounding box
            box_color = (0, 255, 0)  # Verde por defecto
            if contains_similar_chars:
                box_color = (0, 0, 255)  # Rojo si contiene 'Q' o 'O'
                print("Caracteres alfanuméricos similares - Posibilidad alta de error")
            
            # Dibujar un rectángulo alrededor de la matrícula y mostrar el texto detectado
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, text_tesseract, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
            cv2.putText(frame, text_easyocr, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Dibujar un rectángulo semitransparente sobre la ROI
        overlay = frame.copy()
        cv2.rectangle(overlay, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), -1)
        alpha = 0.3  # Transparencia del rectángulo
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Mostrar el fotograma con la matrícula detectada y la ROI sombreada
        cv2.namedWindow(f'Detected License Plate - {video_file}', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f'Detected License Plate - {video_file}', 480, 480)
        cv2.imshow(f'Detected License Plate - {video_file}', frame)

        # Calcular el tiempo de procesamiento del frame
        processing_time = time.time() - start_time
        remaining_time = max(1, int((frame_interval - processing_time) * 1000))

        # Salir al presionar la tecla 'q'
        if cv2.waitKey(remaining_time) & 0xFF == ord('q'):
            break

    # Liberar el objeto de captura de video y cerrar las ventanas de OpenCV
    cap.release()
    cv2.destroyAllWindows()

    print("Procesamiento de video completado.")

# Determinación de velocidad
# Visualización (Premio o castigo)
# Detección de carril (¿?)
def Hilo3():
    for i in range(0, 101, 20):
        print(f"Descarga: {i}% completado")
        time.sleep(1)

# Crear los hilos para cada tarea
hilo1 = threading.Thread(target=Hilo1)
hilo2 = threading.Thread(target=Hilo2)
#hilo3 = threading.Thread(target=Hilo3)

# Iniciar los hilos
hilo1.start()
hilo2.start()
#hilo3.start()

# Esperar a que todos los hilos terminen
hilo1.join()
hilo2.join()
#hilo3.join()

print("Todas las tareas han terminado")