import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import streamlit as st

from sklearn.linear_model    import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_absolute_error, r2_score, mean_squared_error

import analisis as an
from analisis          import convertir_tiempo as hhmmss_a_horas   
from analisis          import horas_a_hhmmss                       
from modelo_regresion  import FEATURES, RUTA_MODELO, ejecutar_pipeline_modelo

warnings.filterwarnings('ignore')

# confguracion 
st.set_page_config(
    page_title="Maratón de Boston 2017",
    layout="wide",
    initial_sidebar_state="expanded"
)


# carga
@st.cache_resource(show_spinner="Cargando modelo de predicción…")
def cargar_modelo():
    return joblib.load(RUTA_MODELO) if os.path.exists(RUTA_MODELO) else None


# evaluacion
@st.cache_data(show_spinner="Re-entrenando modelo…")
def _evaluar_modelo(X_arr, y_arr, test_size):

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=test_size, random_state=42
    )
    modelo = LinearRegression().fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mae_min  = mean_absolute_error(y_test, y_pred) * 60
    rmse_min = np.sqrt(mean_squared_error(y_test, y_pred)) * 60
    r2       = r2_score(y_test, y_pred)

    return mae_min, rmse_min, r2, modelo.coef_, modelo.intercept_

# pestañas
def tab_visualizaciones(df: pd.DataFrame):
    st.header("Retos y Hallazgos — Análisis Visual")
    st.markdown(
        "Exploración de patrones de rendimiento, demografía y comportamiento "
        "de los **26,000+ corredores** del Maratón de Boston 2017."
    )

    with st.spinner("Generando visualizaciones… (puede tardar unos segundos)"):
        todas = an.generar_visualizaciones(df)

    st.markdown("---")
    filtro = st.radio(
        "Filtrar visualizaciones:",
        ["Mostrar Todas", "Solo Retos Técnicos (R)", "Solo Hallazgos (H)"],
        horizontal=True
    )

    if filtro == "Solo Retos Técnicos (R)":
        visualizaciones = [v for v in todas if v["t"].startswith("R")]
    elif filtro == "Solo Hallazgos (H)":
        visualizaciones = [v for v in todas if v["t"].startswith("H")]
    else:
        visualizaciones = todas

    st.markdown("---")
    cols = st.columns(2, gap="large")
    for i, item in enumerate(visualizaciones):
        with cols[i % 2]:
            st.subheader(item["t"])
            st.pyplot(item["f"], use_container_width=True)
            if "desc" in item:
                st.info(item["desc"])
            plt.close(item["f"])
            st.markdown("<br>", unsafe_allow_html=True)


def tab_prediccion(modelo):
    st.header("Predicción de Tiempo Final")
    st.markdown(
        "Introduce tus datos para que el modelo de **Regresión Lineal** "
        "estime tu tiempo de llegada en el Maratón de Boston."
    )

    if modelo is None:
        st.error(
            "No se encontró `modelo_boston.pkl`.\n\n"
            "Sube el CSV para que el modelo se entrene automáticamente."
        )
        return

    # barra lateral
    with st.sidebar:
        st.header("Datos del Corredor")
        st.markdown("Completa todos los campos para obtener la predicción.")
        st.markdown("---")
        st.subheader("Perfil")
        edad       = st.number_input("Edad (años)", min_value=18, max_value=90, value=35)
        genero     = st.selectbox("Género", ["Masculino", "Femenino"])
        genero_num = 0 if genero == "Masculino" else 1
        st.markdown("---")
        st.subheader("Tiempos de Paso")
        st.caption("Formato: H:MM:SS  (ej. 0:22:30)")
        t_5k   = st.text_input("Tiempo en  5 km",   "0:22:00")
        t_10k  = st.text_input("Tiempo en 10 km",   "0:45:00")
        t_half = st.text_input("Tiempo en 21,1 km", "1:35:00")
        t_30k  = st.text_input("Tiempo en 30 km",   "2:18:00")
        st.markdown("---")
        calcular = st.button("Predecir Tiempo Final", use_container_width=True)

    if not calcular:
        st.info("Introduce tus datos en la barra lateral y pulsa **«Predecir Tiempo Final»**.")
        st.markdown("""
        ### ¿Cómo funciona el modelo?
        El simulador usa una **Regresión Lineal** entrenada con los resultados de
        **26,610 corredores** del Maratón de Boston 2017.

        | Variable | Descripción |
        |---|---|
        | `Age` | Edad del corredor en años |
        | `genero_num` | Género (0 = Masculino, 1 = Femenino) |
        | `5K_h` | Tiempo en el km 5 (horas decimales) |
        | `10K_h` | Tiempo en el km 10 (horas decimales) |
        | `Half_h` | Tiempo en el km 21,1 · Media Maratón (horas decimales) |
        | `30K_h` | Tiempo en el km 30 (horas decimales) |
        """)
        return

    etiquetas   = {'5k': t_5k, '10k': t_10k, 'half': t_half, '30k': t_30k}
    convertidos = {k: hhmmss_a_horas(v) for k, v in etiquetas.items()}
    errores     = [f"Tiempo en {k.upper()} no válido"
                   for k, v in convertidos.items() if v is None or np.isnan(v)]

    h_5k, h_10k, h_half, h_30k = convertidos.values()
    if not errores and not (h_5k < h_10k < h_half < h_30k):
        errores.append("Los tiempos deben ser crecientes: 5K < 10K < 21,1K < 30K")

    if errores:
        for e in errores:
            st.error(f"{e}")
        st.info("Corrige los campos marcados y vuelve a pulsar «Predecir».")
        return

    X_input = pd.DataFrame([{
        "Age": edad, "genero_num": genero_num,
        "5K_h": h_5k, "10K_h": h_10k, "Half_h": h_half, "30K_h": h_30k
    }])
    pred_h   = modelo.predict(X_input)[0]
    pred_min = pred_h * 60

    st.success("Predicción completada")
    st.markdown("---")
    col_res, col_det = st.columns([1, 2], gap="large")

    with col_res:
        st.metric(
    label="Tiempo Final Estimado", 
    value=horas_a_hhmmss(pred_h), 
    delta=f"{pred_min:.1f} minutos totales",
    delta_color="off"
)

    with col_det:
        st.subheader("Resumen de la entrada")
        resumen = pd.DataFrame({
            "Variable":        ["Edad", "Género", "5K", "10K", "21,1K", "30K"],
            "Valor original":  [f"{edad} años", genero, t_5k, t_10k, t_half, t_30k],
            "Horas decimales": ["—", "—",
                                f"{h_5k:.4f} h", f"{h_10k:.4f} h",
                                f"{h_half:.4f} h", f"{h_30k:.4f} h"]
        })
        st.dataframe(resumen, hide_index=True, use_container_width=True)
        vel  = 42.195 / pred_h
        pace = pred_min / 42.195
        st.markdown(f"**Velocidad media estimada:** {vel:.2f} km/h  \n"
                    f"**Pace medio estimado:** {pace:.2f} min/km")

    st.markdown("---")
    st.subheader("Contribución de cada variable a la predicción")
    st.caption("Minutos que aporta cada variable al tiempo total predicho, "
               "según los coeficientes del modelo.")

    contrib = modelo.coef_ * X_input.values[0] * 60
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(FEATURES, contrib,
                   color=['green' if c >= 0 else 'red' for c in contrib],
                   edgecolor='white')
    for bar, val in zip(bars, contrib):
        offset = 0.5 if val >= 0 else -0.5
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f'{val:+.1f}', va='center',
                ha='left' if val >= 0 else 'right',
                fontsize=9, fontweight='bold')
    ax.axvline(0, color='black', lw=0.8)
    margen = max(abs(contrib)) * 0.25
    ax.set_xlim(contrib.min() - margen, contrib.max() + margen)
    ax.set(title="Contribución de cada variable al tiempo predicho (minutos)",
           xlabel="Minutos aportados", ylabel="Variable predictora")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")
    st.warning("**Aviso:** Este modelo es una estimación estadística basada en datos históricos "
               "y no garantiza el tiempo exacto en carrera real.")


def tab_evaluacion(df):
    st.header("Evaluación y Entrenamiento del Modelo")
    st.markdown("Ajusta el porcentaje de datos de **prueba**. "
                "El resto se usará para **entrenar**. "
                "Los resultados se cachean: cambiar el slider y volver "
                "al mismo valor no re-entrena el modelo.")

    test_pct = st.slider(
        "Porcentaje para el conjunto de prueba (Test Size)",
        min_value=5, max_value=50, value=20, step=5, format="%d%%"
    )
    test_size = test_pct / 100.0
    st.info(f"El modelo se entrenará con el **{100 - test_pct}%** de los corredores "
            f"y se pondrá a prueba con el **{test_pct}%** restante.")

    df_model = df.dropna(subset=FEATURES + ["Official_Time_h"])
    X_arr = df_model[FEATURES].to_numpy()
    y_arr = df_model["Official_Time_h"].to_numpy()

    mae_min, rmse_min, r2, coefs, intercept = _evaluar_modelo(X_arr, y_arr, test_size)

    st.markdown("---")
    st.subheader("Métricas de Rendimiento")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Error Absoluto Medio (MAE)", f"{mae_min:.2f} min",
                  help="Promedio de minutos de error respecto al tiempo real.")
    with col2:
        st.metric("Variabilidad / RMSE", f"{rmse_min:.2f} min",
                  help="Penaliza más los errores grandes. Mide la variabilidad de la predicción.")
    with col3:
        st.metric("R² (Coef. de Determinación)", f"{r2:.4f}",
                  help="1.0 = modelo perfecto. Mide qué fracción de la varianza explica el modelo.")

# barra lateral global
def barra_lateral_global(df):
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Estadísticas del Dataset**")
        st.metric("Total de corredores", f"{len(df):,}")
        if 'M/F' in df.columns:
            dist = df['M/F'].value_counts()
            st.metric("Hombres", f"{dist.get('M', 0):,}")
            st.metric("Mujeres", f"{dist.get('F', 0):,}")
        if 'Age' in df.columns:
            st.metric("Edad media", f"{df['Age'].mean():.1f} años")
        if 'Official_Time_h' in df.columns:
            st.metric("Tiempo mediano",
                      horas_a_hhmmss(df['Official_Time_h'].median()))

def main():
    with st.sidebar:
        st.title("Maratón de Boston 2017")
        st.caption("Programación II · Práctica 2")

    with st.spinner("Limpiando y analizando datos..."):
        # Lee el CSV directamente desde la carpeta del repositorio
        df_bruto = pd.read_csv("marathon_results_2017.csv")
        df = an.cargar_y_limpiar(df_bruto) 

    if not os.path.exists(RUTA_MODELO):
        with st.spinner("Entrenando modelo de predicción…"):
            modelo = ejecutar_pipeline_modelo(df)
    else:
        modelo = cargar_modelo()

    barra_lateral_global(df)

    tab1, tab2, tab3 = st.tabs([" Visualizaciones", "Evaluación",  "Predicción"])

    with tab1:
        tab_visualizaciones(df)
    with tab2:
        tab_evaluacion(df)
    with tab3:
        tab_prediccion(modelo)
        
        
if __name__ == "__main__":
    main()
    
    
