# Pruebas-GRB

Este repositorio incluye un script en Python para generar pronósticos de tickets y ventas diarias del Restaurante Kuche a partir de un archivo de Excel con datos históricos.

## Requisitos

- Python 3.9+
- pandas
- numpy
- scikit-learn
- matplotlib
- xlsxwriter (se instala automáticamente con pandas, pero se puede asegurar con `pip install XlsxWriter`)

Puedes instalar las dependencias ejecutando:

```bash
pip install pandas numpy scikit-learn matplotlib XlsxWriter
```

## Uso

1. Ubica el archivo original de datos (por ejemplo `BD Tickets para predicciones.xlsx`).
2. Ejecuta el script indicando la ruta al archivo:

```bash
python predicciones_restaurante.py "C:\\Users\\gonza\\Dropbox\\1 Documentos 2025\\1 Nesswork 2025\\0 Proyectos Nesswork 2025\\Kuchen 2025\\BD Tickets para predicciones.xlsx"
```

Opcionalmente, puedes indicar el número de días a pronosticar (por defecto son 15):

```bash
python predicciones_restaurante.py "ruta/al/archivo.xlsx" --forecast_days 30
```

## Resultados

Al ejecutar el script se generarán:

- Un archivo de Excel llamado `Predicciones Restaurante Kuche [fecha] [hora].xlsx` en el mismo directorio del archivo original con:
  - Serie histórica.
  - Validación con predicciones sobre el conjunto de prueba.
  - Pronósticos para los próximos días.
  - Métricas de precisión (MAE, RMSE y R²).
- Dos gráficos en formato PNG con la evolución histórica y los pronósticos de tickets y ventas.
- Un resumen de las métricas mostrado en la consola.

Si el archivo de origen contiene pequeñas variaciones en los nombres de las columnas (por ejemplo `Ticekts` o `Venta Diaria`), el script las normaliza automáticamente.
