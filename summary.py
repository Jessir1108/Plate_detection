import csv
import os
from utils import select_csv_file

# Especificar la carpeta que contiene los archivos CSV de placas conocidas y resultados
known_plates_folder = 'known_plates'
results_folder = 'results'
summaries_folder = 'summaries'

# Crear la carpeta summaries si no existe
if not os.path.exists(summaries_folder):
    os.makedirs(summaries_folder)

# Seleccionar el archivo CSV de placas conocidas
known_csv_path = select_csv_file(known_plates_folder, "Seleccione el archivo CSV de placas conocidas:")

# Seleccionar el archivo CSV de resultados
results_csv_path = select_csv_file(results_folder, "Seleccione el archivo CSV de resultados:")

# Solicitar al usuario que ingrese el nombre del archivo CSV de resumen
summary_csv_name = input("Ingrese el nombre del archivo CSV de resumen (sin extensión): ")
summary_csv_path = os.path.join(summaries_folder, f"{summary_csv_name}.csv")

# Leer placas conocidas
known_plates = set()
total_plates = 0
discriminated_plates = 0
plates_with_q_or_o = set()
with open(known_csv_path, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        plate = row['plate']
        known_plates.add(plate)
        total_plates += 1
        if 'O' in plate or 'Q' in plate:
            discriminated_plates += 1
            plates_with_q_or_o.add(plate)

# Solicitar al usuario el número de placas en mal estado
bad_status_plates = int(input("Ingrese el número de placas en mal estado que no se deben tener en cuenta: "))
discriminated_plates += bad_status_plates

# Leer resultados
results = []
with open(results_csv_path, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        results.append(row)

# Inicializar contadores y conjuntos para detecciones únicas
tesseract_detections = set()
easyocr_detections = set()
tesseract_q_or_o_detections = set()
easyocr_q_or_o_detections = set()

# Contar detecciones únicas
for result in results:
    tesseract_text = result['Tesseract Text']
    easyocr_text = result['EasyOCR Text']
    
    if tesseract_text in known_plates:
        tesseract_detections.add(tesseract_text)
        if tesseract_text in plates_with_q_or_o:
            tesseract_q_or_o_detections.add(tesseract_text)
    
    if easyocr_text in known_plates:
        easyocr_detections.add(easyocr_text)
        if easyocr_text in plates_with_q_or_o:
            easyocr_q_or_o_detections.add(easyocr_text)

# Calcular porcentajes de éxito
average_success_all_tesseract = len(tesseract_detections) / total_plates * 100
average_success_all_easyocr = len(easyocr_detections) / total_plates * 100

# Calcular porcentajes de éxito con discriminación
valid_plates = total_plates - discriminated_plates
average_success_discriminated_tesseract = len(tesseract_detections) / valid_plates * 100 if valid_plates > 0 else 0
average_success_discriminated_easyocr = len(easyocr_detections) / valid_plates * 100 if valid_plates > 0 else 0

# Calcular precisión para placas con 'Q' u 'O'
total_q_or_o_plates = len(plates_with_q_or_o)
precision_q_or_o_tesseract = len(tesseract_q_or_o_detections) / total_q_or_o_plates * 100 if total_q_or_o_plates > 0 else 0
precision_q_or_o_easyocr = len(easyocr_q_or_o_detections) / total_q_or_o_plates * 100 if total_q_or_o_plates > 0 else 0

# Generar resumen
summary_data = [
    ['OCR System', 'Unique Detections', 'Amount of Plates', 'Discriminated Plates', 'Average of Success (All)', 'Average of Success with Discrimination', 'Precision with Q or O'],
    ['Tesseract', len(tesseract_detections), total_plates, discriminated_plates, f"{average_success_all_tesseract:.2f}%", f"{average_success_discriminated_tesseract:.2f}%", f"{precision_q_or_o_tesseract:.2f}%"],
    ['EasyOCR', len(easyocr_detections), total_plates, discriminated_plates, f"{average_success_all_easyocr:.2f}%", f"{average_success_discriminated_easyocr:.2f}%", f"{precision_q_or_o_easyocr:.2f}%"]
]

# Escribir resumen en el archivo CSV
with open(summary_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(summary_data)

print("Resumen generado en:", summary_csv_path)