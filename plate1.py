import yolov5
import cv2
import pytesseract
import easyocr
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# Cargar el modelo preentrenado de YOLOv5 para la detección de matrículas
model = yolov5.load('keremberke/yolov5m-license-plate')

# Especificar la carpeta que contiene las imágenes
images_folder = '.'

# Obtener la lista de archivos de imagen en la carpeta
image_files = [f for f in os.listdir(images_folder) if f.endswith('.png')]
if not image_files:
    print("No se encontró ningún archivo de imagen en la carpeta.")
    exit()

# Listar las imágenes con números
print("Seleccione la imagen que desea procesar:")
for idx, image_file in enumerate(image_files, start=1):
    print(f"Imagen {image_file} - {idx}")

# Solicitar al usuario que seleccione una imagen
image_index = int(input("Ingrese el número de la imagen que desea procesar: ")) - 1
if image_index < 0 or image_index >= len(image_files):
    print("Número de imagen inválido.")
    exit()

image_file = image_files[image_index]
image_path = os.path.join(images_folder, image_file)

# Configuración personalizada para Tesseract
custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# Inicializar el lector de EasyOCR
reader = easyocr.Reader(['en'])

# Cargar la imagen
image = cv2.imread(image_path)

# Verificar si la imagen se cargó correctamente
if image is None:
    print(f"Error: No se pudo abrir el archivo de imagen {image_file}")
    exit()

# Aumentar el brillo
brightness = 50
bright_image = cv2.convertScaleAbs(image, alpha=1, beta=brightness)

# Aumentar el contraste
contrast = 1.5
contrast_image = cv2.convertScaleAbs(bright_image, alpha=contrast, beta=0)

# Aumentar la nitidez
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharp_image = cv2.filter2D(contrast_image, -1, kernel)

# Convertir la imagen a escala de grises
gray_image = cv2.cvtColor(sharp_image, cv2.COLOR_BGR2GRAY)

# Realizar la detección en la imagen original
results = model(image)

# Crear una figura con subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Función para aplicar OCR y dibujar resultados
def apply_ocr_and_draw(image, ax, title):
    # Aplicar OCR con Tesseract
    text_tesseract = pytesseract.image_to_string(image, config=custom_config).strip()
    
    # Aplicar OCR con EasyOCR
    result_easyocr = reader.readtext(image, detail=0)
    text_easyocr = result_easyocr[0] if result_easyocr else ""
    
    # Dibujar resultados en la imagen
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, text_tesseract, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(image, text_easyocr, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Mostrar la imagen en el subplot
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis('off')
    print(f'{title} - Tesseract detectó: {text_tesseract}')
    print(f'{title} - EasyOCR detectó: {text_easyocr}')

# Aplicar OCR y dibujar resultados en cada etapa
apply_ocr_and_draw(image.copy(), axs[0, 0], 'Imagen Original')
apply_ocr_and_draw(bright_image.copy(), axs[0, 1], 'Aumento de Brillo')
apply_ocr_and_draw(sharp_image.copy(), axs[1, 0], 'Aumento de Nitidez')
apply_ocr_and_draw(gray_image.copy(), axs[1, 1], 'Escala de Grises')

# Ajustar el diseño y mostrar la figura con subplots
plt.tight_layout()
plt.show()

print("Procesamiento de imagen completado.")