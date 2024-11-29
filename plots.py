import matplotlib.pyplot as plt

# Datos para el diagrama de líneas
etiquetas = [1, 2, 3, 4, 5, 6]
reales = [49, 49, 49, 47, 49, 48]
estimadas = [55, 46, 55, 53, 51, 47]

# Crear la gráfica
plt.figure(figsize=(10, 5), facecolor='none')

# Graficar las líneas con un grosor mayor
plt.plot(etiquetas, reales, label='Reales', color='#41b8d5', marker='o', linewidth=3)
plt.plot(etiquetas, estimadas, label='Estimadas', color='#e7c24d', marker='o', linewidth=3)

# Añadir leyendas y títulos
plt.legend(fontsize=12)
plt.xlabel('Vehículo', fontsize=20, fontweight='bold')
plt.ylabel('Velocidad (Km/h)', fontsize=20, fontweight='bold')
plt.title('Valores Reales vs Estimados', fontsize=20, fontweight='bold')

# Ajustar los intervalos del eje y a cada 5 km/h
plt.yticks(range(30, 60, 5), fontsize=12, fontweight='bold')

# Fondo transparente
plt.gca().patch.set_alpha(0)

# Mostrar la gráfica
plt.tight_layout()
plt.show()
