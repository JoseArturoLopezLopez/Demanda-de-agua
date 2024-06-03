import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

# Cargar datos
ruta_datos = 'Demanda de agua.csv'  # Asegúrate de actualizar la ruta
datos = pd.read_csv(ruta_datos, usecols=[1], dtype=np.float64)  # Índice de la segunda columna es 1
datos = datos.values.flatten()  # Convertir a un array de NumPy

# Normalizar los datos
max_valor = np.max(datos)
min_valor = np.min(datos)
datos_normalizados = (datos - min_valor) / (max_valor - min_valor)

# Función para crear secuencias
def crear_secuencias(datos, longitud_secuencia):
    entradas, salidas = [], []
    for i in range(len(datos) - longitud_secuencia):
        entrada = datos[i:(i + longitud_secuencia)]
        salida = datos[i + longitud_secuencia]
        entradas.append(entrada.reshape((longitud_secuencia, 1)))  # Añadir dimensión de características
        salidas.append(salida)
    return np.array(entradas), np.array(salidas)

longitud_secuencia = 20
X, y = crear_secuencias(datos_normalizados, longitud_secuencia)

# Mostrando las primeras 20 secuencias
for i in range(20):
    print(f"Secuencia {i + 1}: \n{X[i]} -> Salida: {y[i]}")

# Dividir los datos en entrenamiento y prueba
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir el modelo LSTM
modelo = Sequential([
    LSTM(30, input_shape=(longitud_secuencia, 1), return_sequences=True),
    LSTM(30, return_sequences=False),
    Dense(1)
])

modelo.compile(optimizer='adam', loss='mean_squared_error')

historial = modelo.fit(X_entrenamiento, y_entrenamiento, epochs=200, batch_size=64)

# Realizar predicciones
predicciones = modelo.predict(X_prueba)

# Calcular el Mean Squared Error
mse = mean_squared_error(y_prueba, predicciones)
print("Mean Squared Error:", mse)

# Graficar los resultados
plt.figure(figsize=(12, 6))
plt.plot(y_prueba, label='Datos Reales', marker='o')  # Usar marcadores para claridad
plt.plot(predicciones.flatten(), label='Predicción del Modelo', marker='*', linestyle='dotted')  # Asegurar que las predicciones estén en la misma forma que los datos reales
plt.title('Comparación de los Datos Reales con las Predicciones del Modelo LSTM')
plt.xlabel('Número de Muestra')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)  # Agregar una grilla para mejor visualización
plt.show()