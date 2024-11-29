import yolov5
import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re

# Forzar el uso del backend TkAgg
matplotlib.use('TkAgg')

# Cargar el modelo preentrenado de YOLOv5 para la detección de matrículas
model = yolov5.load('keremberke/yolov5m-license-plate')

# Cargar la imagen
image_path = 'plate_detection.png'
image = cv2.imread(image_path)

# Configuración personalizada para Tesseract
custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# Expresión regular para validar el patrón LLLNNN
pattern = re.compile(r'^[A-Z]{3}[0-9]{3}$')

# Realizar la detección en la imagen
results = model(image)

# Procesar la primera matrícula detectada en la imagen
for result in results.xyxy[0]:
    x1, y1, x2, y2, conf, cls = result
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Recortar la región de la matrícula
    plate_region = image[y1:y2, x1:x2]
    
    # OCR en la imagen original recortada
    text_original = pytesseract.image_to_string(plate_region, config=custom_config).strip()
    if pattern.match(text_original):
        print(f'OCR en Recorte Original: {text_original}')
    
    # Convertir la imagen recortada a escala de grises
    gray_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
    
    # OCR en la imagen en escala de grises
    text_gray = pytesseract.image_to_string(gray_plate, config=custom_config).strip()
    if pattern.match(text_gray):
        print(f'OCR en Gray Scale: {text_gray}')
    
    # Reducir el ruido usando Gaussian Blur
    denoised_plate = cv2.GaussianBlur(gray_plate, (3, 5), .9)    
    # OCR en la imagen después de reducción de ruido
    text_denoised = pytesseract.image_to_string(denoised_plate, config=custom_config).strip()
    if pattern.match(text_denoised):
        print(f'OCR en Noise Reduction: {text_denoised}')
    
    # Aplicar umbralado para binarizar la imagen
    binary_plate = cv2.adaptiveThreshold(denoised_plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
    
    # # Aplicar dilatación y erosión
    # kernel = np.ones((3, 3), np.uint8)
    # dilated_plate = cv2.dilate(binary_plate, kernel, iterations=1)
    # eroded_plate = cv2.erode(dilated_plate, kernel, iterations=1)
    
    # OCR en la imagen binarizada y procesada
    text_processed = pytesseract.image_to_string(binary_plate, config=custom_config).strip()
    print(f'OCR en Binarization + Dilation + Erosion: {text_processed}')
    
    # Aumentar el brillo
    brightness = 50
    bright_image = cv2.convertScaleAbs(plate_region, alpha=1, beta=brightness)
    text_bright = pytesseract.image_to_string(bright_image, config=custom_config).strip()
    print(f'OCR en Aumento de Brillo: {text_bright}')
    
    # Aumentar el contraste
    contrast = 1.5
    contrast_image = cv2.convertScaleAbs(bright_image, alpha=contrast, beta=0)
    text_contrast = pytesseract.image_to_string(contrast_image, config=custom_config).strip()
    print(f'OCR en Aumento de Contraste: {text_contrast}')
    
    # Aumentar la saturación
    hsv_image = cv2.cvtColor(plate_region, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 1] = cv2.add(hsv_image[:, :, 1], 50)
    saturated_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    text_saturated = pytesseract.image_to_string(saturated_image, config=custom_config).strip()
    print(f'OCR en Aumento de Saturación: {text_saturated}')
    
    # Crear un layout 3x3 con las imágenes
    layout_image = np.zeros((plate_region.shape[0] * 3, plate_region.shape[1] * 3, 3), dtype=np.uint8)

    # Recorte Original
    layout_image[:plate_region.shape[0], :plate_region.shape[1], :] = cv2.cvtColor(plate_region, cv2.COLOR_BGR2RGB)

    # Gray Scale (convertido a BGR para que tenga 3 canales)
    gray_bgr = cv2.cvtColor(gray_plate, cv2.COLOR_GRAY2BGR)
    layout_image[:plate_region.shape[0], plate_region.shape[1]:plate_region.shape[1]*2, :] = gray_bgr

    # Noise Reduction (convertido a BGR para que tenga 3 canales)
    denoised_bgr = cv2.cvtColor(denoised_plate, cv2.COLOR_GRAY2BGR)
    layout_image[:plate_region.shape[0], plate_region.shape[1]*2:, :] = denoised_bgr

    # Binarization + Dilation + Erosion (convertido a BGR para que tenga 3 canales)
    processed_bgr = cv2.cvtColor(binary_plate, cv2.COLOR_GRAY2BGR)
    layout_image[plate_region.shape[0]:plate_region.shape[0]*2, :plate_region.shape[1], :] = processed_bgr

    # Aumento de Brillo
    layout_image[plate_region.shape[0]:plate_region.shape[0]*2, plate_region.shape[1]:plate_region.shape[1]*2, :] = cv2.cvtColor(bright_image, cv2.COLOR_BGR2RGB)

    # Aumento de Contraste
    layout_image[plate_region.shape[0]:plate_region.shape[0]*2, plate_region.shape[1]*2:, :] = cv2.cvtColor(contrast_image, cv2.COLOR_BGR2RGB)

    # Aumento de Saturación
    layout_image[plate_region.shape[0]*2:, :plate_region.shape[1], :] = cv2.cvtColor(saturated_image, cv2.COLOR_BGR2RGB)

    # Mostrar la imagen del layout usando plt.imshow
    plt.figure(figsize=(15, 15))
    plt.imshow(layout_image)
    plt.title('Recorte Original, Gray Scale, Noise Reduction, Binarization + Dilation + Erosion, Aumento de Brillo, Aumento de Contraste, Aumento de Saturación')
    plt.axis('off')
    plt.show()

print("Procesamiento de la imagen completado.")