import threading
import time
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
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

# Cargar el modelo preentrenado de YOLOv5 para la detección de matrículas
model = yolov5.load('keremberke/yolov5m-license-plate')

# Especificar la carpeta que contiene los videos
videos_folder = 'videos'

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
