import pandas as pd

# Initialize global metrics
total_plates_filtered = 0
correct_detections_filtered = 0

# Process each pair of files
for i in range(1, 6):
    plates_df = pd.read_csv(f'known_plates/plates_prueba{i}.csv')
    results_df = pd.read_csv(f'results/prueba{i}_savimg_results.csv')

    # Handle NaN values in 'plate' column
    plates_df['plate'] = plates_df['plate'].fillna('')

    for _, plate_row in plates_df.iterrows():
        plate = plate_row['plate']
        match_found_filtered = False

        if 'Q' not in plate and 'O' not in plate:
            for _, result_row in results_df.iterrows():
                detected_plate = str(result_row['EasyOCR Text']) if not pd.isna(result_row['EasyOCR Text']) else ''  # Handle NaN

                # Verificar si hay coincidencia exacta
                if detected_plate == plate:
                    match_found_filtered = True
                    break  # Detener la búsqueda una vez encontrada la coincidencia

            if match_found_filtered:
                correct_detections_filtered += 1

            total_plates_filtered += 1

# Calcular la Exactitud Binaria
binary_accuracy_filtered = (correct_detections_filtered / total_plates_filtered) * 100 if total_plates_filtered > 0 else 0

# Calcular la Tasa de Falsos Negativos
false_negatives_filtered = total_plates_filtered - correct_detections_filtered
false_negative_rate_filtered = (false_negatives_filtered / total_plates_filtered) * 100 if total_plates_filtered > 0 else 0

print(f'Binary Accuracy per Plate (Excluding Q and O): {binary_accuracy_filtered:.2f}%')
print(f'False Negative Rate (Excluding Q and O): {false_negative_rate_filtered:.2f}%')
print(f'Total Plates (Excluding Q and O): {total_plates_filtered}')
print(f'Correct Detections (Excluding Q and O): {correct_detections_filtered}')
print(f'False Negatives (Excluding Q and O): {false_negatives_filtered}')