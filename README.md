# PROYECTO: MARATÓN DE BOSTON 2017 - ANÁLISIS Y PREDICCIÓN


## Descripción
Este proyecto es una aplicación web interactiva desarrollada con Streamlit que analiza los datos de los más de 26,000 corredores que participaron en la Maratón de Boston 2017. El sistema no solo realiza un análisis exploratorio visual profundo de los datos, sino que también implementa un modelo de Machine Learning (Regresión Lineal) para predecir el tiempo final de un corredor basándose en sus características físicas y sus tiempos parciales.

## Estructura de Archivos
El proyecto se compone de los siguientes scripts principales:

* `dashboard.py`: Es el archivo principal que levanta la interfaz gráfica (Streamlit). Gestiona la navegación por pestañas y la barra lateral de configuración y predicción.
* `analisis.py`: Módulo que actúa como motor de procesamiento de datos. Se encarga de cargar y limpiar el dataset original (transformando formatos de tiempo a horas decimales, calculando velocidades, etc.) y contiene la lógica para generar todos los gráficos descriptivos y hallazgos utilizando Matplotlib y Seaborn.
* `modelo_regresion.py`: Define el pipeline completo de Machine Learning. Entrena un modelo de regresión lineal seleccionando variables clave (Edad, Género, tiempos en 5K, 10K, Media Maratón y 30K), calcula métricas de error y guarda el modelo entrenado como `modelo_boston.pkl` para evitar tener que reentrenarlo en cada ejecución.
* `marathon_results_2017.csv`: El conjunto de datos original.

## Características de la Aplicación Web
Al iniciar el dashboard, la web muestra una barra lateral persistente con estadísticas globales del dataset (participantes totales, desglose por género, tiempo mediano, etc.). El contenido principal se divide en tres pestañas:

1. Visualizaciones:
   Muestra un extenso catálogo de gráficos divididos en "Retos Técnicos (R)" y "Hallazgos (H)". Se exploran temas como la degradación del ritmo ("El muro"), el impacto de un inicio rápido, la distribución de edades y las categorías competitivas. Incluye un filtro en la parte superior para facilitar la navegación.

2. Evaluación:
   Una sección interactiva que expone las métricas de rendimiento del modelo predictivo (Error Absoluto Medio - MAE, RMSE y R²). Permite al usuario usar un slider para cambiar dinámicamente el porcentaje de datos destinados al testeo (Test Size) y observar cómo varían estas métricas al reentrenarse el modelo.

3. Predicción:
   Un simulador donde el usuario introduce sus propios datos a través de la barra lateral (Edad, Género, y tiempos esperados en 5K, 10K, 21.1K y 30K). La aplicación estima el tiempo de llegada final y la velocidad media
   . Además, genera un gráfico de barras explicativo que detalla cuántos minutos positivos o negativos aporta cada variable al cálculo final.

## Requisitos de Instalación
Para poder ejecutar el código, asegúrate de tener instalado Python y las siguientes dependencias:
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

Puedes instalarlas usando pip:
pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib

## Cómo Ejecutar el Proyecto
1. Asegúrate de tener el archivo `marathon_results_2017.csv` en el mismo directorio que los scripts `.py`.
2. Abre una terminal o línea de comandos en esa misma carpeta.
3. Ejecuta el siguiente comando:
