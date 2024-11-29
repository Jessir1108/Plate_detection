import os
import pandas as pd

# Ruta de la carpeta
folder_path = 'known_plates'

# Inicializar contadores
total_plates = 0
plates_with_q_or_o = 0

# Iterar sobre todos los archivos en la carpeta
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        # Leer el archivo CSV
        df = pd.read_csv(file_path)
        # Contar el número total de placas
        total_plates += len(df)
        # Filtrar y contar las placas que contienen 'Q' u 'O'
        filtered_plates = df['plate'].str.contains('Q|O', case=False)
        plates_with_q_or_o += filtered_plates.sum()

# Calcular el porcentaje
percentage = (plates_with_q_or_o / total_plates) * 100 if total_plates > 0 else 0

# Imprimir los resultados
print(f'Número total de placas: {total_plates}')
print(f'Número de placas que contienen Q u O: {plates_with_q_or_o}')
print(f'Porcentaje de placas que contienen Q u O: {percentage:.2f}%')