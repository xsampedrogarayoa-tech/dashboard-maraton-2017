import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib

from sklearn.linear_model    import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

RUTA_MODELO = "modelo_boston.pkl"

FEATURES = ["Age", "genero_num", "5K_h", "10K_h", "Half_h", "30K_h"]
TARGET   = "Official_Time_h"


# division train/test
def preparar_datos(df, test_size = 0.2):
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=42)


# entrenamiento
def entrenar_modelo(X_train, y_train):
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    return modelo


# métricas
def calcular_metricas(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    return {
        "mae_min":  mean_absolute_error(y_test, y_pred) * 60,
        "rmse_min": np.sqrt(mean_squared_error(y_test, y_pred)) * 60,
        "r2":       r2_score(y_test, y_pred),
    }


# importancia de cada variable
def graficar_importancia(modelo, features):
    coefs_min = modelo.coef_ * 60

    importancia = (pd.DataFrame({'Variable': features, 'Coeficiente': coefs_min})
                   .sort_values('Coeficiente', key=abs, ascending=True))

    colores = ['green' if v >= 0 else 'red' for v in importancia['Coeficiente']]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(importancia['Variable'], importancia['Coeficiente'],
                   color=colores, edgecolor='white', height=0.6)

    for bar, val in zip(bars, importancia['Coeficiente']):
        offset = 0.3 if val >= 0 else -0.3
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f'{val:+.2f} min/u',
                va='center', ha='left' if val >= 0 else 'right', fontsize=9)

    ax.axvline(0, color='black', lw=0.8, ls='--')
    ax.set_title('Influencia de cada Variable en la Predicción del Tiempo Final\n'
                 '(Coeficientes de Regresión Lineal — minutos por unidad)',
                 fontsize=12)
    ax.set_xlabel('Coeficiente (min de variación en la predicción)', fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.margins(x=0.3)

    leyenda = [
        mpatches.Patch(facecolor='green', label='Aumenta el tiempo predicho'),
        mpatches.Patch(facecolor='red', label='Reduce el tiempo predicho'),
    ]
    ax.legend(handles=leyenda, loc='lower right', fontsize=9)
    plt.tight_layout()

    plt.close(fig)
    return fig


# exportar
def exportar_modelo(modelo, ruta):
    joblib.dump(modelo, ruta)



# pipeline 
def ejecutar_pipeline_modelo(df):
    df_modelo = df[FEATURES + [TARGET]].dropna()

    X_train, X_test, y_train, y_test = preparar_datos(df_modelo)

    modelo = entrenar_modelo(X_train, y_train)

    graficar_importancia(modelo, FEATURES)

    exportar_modelo(modelo, RUTA_MODELO)

    return modelo