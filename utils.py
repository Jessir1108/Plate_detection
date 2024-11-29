# utils.py
import os

def select_csv_file(folder_path, prompt):
    """
    Lista los archivos CSV en una carpeta y permite al usuario seleccionar uno.

    Args:
        folder_path (str): La ruta de la carpeta que contiene los archivos CSV.
        prompt (str): El mensaje que se mostrará al usuario para seleccionar un archivo.

    Returns:
        str: La ruta completa del archivo CSV seleccionado.
    """
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not csv_files:
        print(f"No se encontró ningún archivo CSV en la carpeta {folder_path}.")
        exit()

    print(prompt)
    for idx, csv_file in enumerate(csv_files, start=1):
        print(f"{idx}. {csv_file}")

    csv_index = int(input("Ingrese el número del archivo CSV que desea procesar: ")) - 1
    if csv_index < 0 or csv_index >= len(csv_files):
        print("Número de archivo CSV inválido.")
        exit()

    return os.path.join(folder_path, csv_files[csv_index])