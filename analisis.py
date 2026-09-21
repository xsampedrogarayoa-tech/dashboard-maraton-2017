import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings('ignore')

# segmentos seleccionados para el reto 13 y hallazgo 2
SEG_COLS = [
    'seg_0_5K', 'seg_5_10K', 'seg_10_15K', 'seg_15_20K',
    'seg_20_25K', 'seg_25_30K', 'seg_30_35K', 'seg_35_40K'
]
ETQ_SEG = ['0–5K', '5–10K', '10–15K', '15–20K',
           '20–25K', '25–30K', '30–35K', '35–40K']

# conversores
def convertir_tiempo(t):
    if pd.isna(t) or str(t).strip() in ('-', '', 'nan', 'NaN'):
        return np.nan
    try:
        partes = str(t).strip().split(':')
        if len(partes) == 3:
            h, m, s = float(partes[0]), float(partes[1]), float(partes[2])
            return h + m / 60.0 + s / 3600.0
        if len(partes) == 2:
            m, s = float(partes[0]), float(partes[1])
            return m / 60.0 + s / 3600.0
        return np.nan
    except (ValueError, TypeError):
        return np.nan


def horas_a_hhmmss(horas):
    if horas is None or pd.isna(horas):
        return '--:--:--'
    total_seg = int(round(horas * 3600))
    h = total_seg // 3600
    m = (total_seg % 3600) // 60
    s = total_seg % 60
    return f"{h}:{m:02d}:{s:02d}"


def pace_milla_a_seg_km(t):
    val = convertir_tiempo(t)
    return np.nan if pd.isna(val) else (val * 3600) / 1.60934


# formato para ahorrarme repetirlo
def _fmt_tiempo(ax, which='x'):
    fmt = mticker.FuncFormatter(lambda x, _: f"{int(x // 60)}h {int(x % 60)}m")
    (ax.xaxis if which == 'x' else ax.yaxis).set_major_formatter(fmt)


# carga y limpieza
def cargar_y_limpiar(df):
    df = df.copy()
    df.drop(columns=[c for c in ['Unnamed: 0', 'Unnamed: 9', 'Citizen', 'Proj Time']
                     if c in df.columns], inplace=True)

    # checkpoints y tiempo final a horas decimales
    for col in ['5K', '10K', '15K', '20K', 'Half', '25K', '30K', '35K', '40K', 'Official Time']:
        df[col.replace(' ', '_') + '_h'] = df[col].apply(convertir_tiempo)

    df['Pace_seg_km']   = df['Pace'].apply(pace_milla_a_seg_km)
    df['genero_num']    = df['M/F'].map({'M': 0, 'F': 1})
    df['velocidad_kmh'] = 42.195 / df['Official_Time_h']

    chk_abs   = ['5K_h', '10K_h', '15K_h', '20K_h',
                 '25K_h', '30K_h', '35K_h', '40K_h', 'Official_Time_h']
    seg_names = SEG_COLS + ['seg_40_fin']
    tmp = df[chk_abs].diff(axis=1)
    tmp.iloc[:, 0] = df['5K_h']        # defino el primer segmento como tiempo absoluto en 5K
    tmp.columns    = seg_names
    df[seg_names]  = tmp

    # ritmo por mitad y diferencia
    df['pace_1a_mitad_min_km'] = (df['Half_h'] * 60) / 21.0975
    df['pace_2a_mitad_min_km'] = ((df['Official_Time_h'] - df['Half_h']) * 60) / 21.0975
    df['ritmo_diferencia']     = df['pace_2a_mitad_min_km'] - df['pace_1a_mitad_min_km']
    df['categoria']            = np.where(df['Official_Time_h'] < 2.5, 'Élite (<2.5h)', 'Amateur (≥2.5h)')
    df['Official_Time_min']    = df['Official_Time_h'] * 60

    cols_clave = ['Official_Time_h', 'Age', 'genero_num',
                  '5K_h', '10K_h', '15K_h', '20K_h',
                  'Half_h', '25K_h', '30K_h', '35K_h', '40K_h']
    df.dropna(subset=cols_clave, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# visualizaciones
def generar_visualizaciones(df):
    vis = []

    def add(titulo: str, fig, desc: str):
        plt.tight_layout()
        vis.append({"t": titulo, "f": fig, "desc": desc})

    # R-1  Distribución de Edad 
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['Age'], bins=40, color='#1976D2', alpha=0.75, kde=True,
                 line_kws={'color': '#0D47A1', 'lw': 2}, ax=ax)
    ax.axvline(df['Age'].median(), color='#FDD835', lw=2, ls='--',
               label=f'Mediana: {df["Age"].median():.0f} años')
    ax.set(xlabel='Edad (años)', ylabel='Número de corredores')
    ax.legend(fontsize=9)
    add("R-1 Distribución de Edad", fig,
        f"La concentración principal se encuentra entre 30 y 50 años y vemos como la mediana se sitúa en {df['Age'].median():.0f} años. Como curiosidad, hallamos en perfiles extremos: de {df['Age'].min()} y {df['Age'].max()} años.")

    # R-2  Distribución de Género 
    conteos = df['M/F'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    _, _, textos_r2 = ax.pie(
        [conteos.get('M', 0), conteos.get('F', 0)],
        labels=[f"Masculino\n{conteos.get('M',0):,}", f"Femenino\n{conteos.get('F',0):,}"],
        colors=['blue', 'red'], autopct='%1.1f%%',
        wedgeprops=dict(width=0.6, edgecolor='white'), textprops={'fontsize': 12}
    )
    plt.setp(textos_r2, color='white', weight='bold')
    add("R-2 Distribución de Género", fig,
        f"La participación masculina de un {conteos.get('M', 0) / len(df) * 100:.1f}% es ligeramente superior a el {conteos.get('F', 0) / len(df) * 100:.1f}% femenino pero vemos que ambos grupos estan equilibrados.")

    # R-3  Tiempos por Género 
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=df, x='M/F', y='Official_Time_min',
                palette={'M': 'blue', 'F': 'red'}, notch=True, ax=ax)
    ax.set(xlabel='Género', ylabel='Tiempo oficial')
    ax.set_xticklabels(['Masculino', 'Femenino'])
    _fmt_tiempo(ax, 'y')
    add("R-3 Tiempos por Género", fig,
        "Los hombres completan el maratón con una mediana aproximadamente 20 min inferior. Pero encontramos una dispersión similar en ambos géneros, resaltan las outliers mujer por debajo de 2 h 40 min.")

    #  R-4  Top 10 Países 
    top10 = df['Country'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=top10.values[::-1], y=top10.index[::-1],
                palette=sns.color_palette("Blues_d", 10), ax=ax)
    ax.set(xlabel='Número de corredores', ylabel='País')
    add("R-4 Top 10 Países", fig,
        f"En esta gráfica EE.UU. domina claramente con {top10.iloc[0]:,} participantes, y le siguen Canadá con {top10.iloc[1]:,}, Gran Bretaña con {top10.iloc[2]:,} y México con {top10.iloc[3]:,}.")

    #  R-5  Correlación de Tiempos Parciales 
    cols_c = ['5K_h', '10K_h', '20K_h', 'Half_h', '30K_h', '40K_h', 'Official_Time_h']
    etqr5  = ['5K', '10K', '20K', 'Media', '30K', '40K', 'Final']
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df[cols_c].corr(),
                mask=np.triu(np.ones((7, 7), bool)),
                annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.9, vmax=1.0,
                xticklabels=etqr5, yticklabels=etqr5, linewidths=0.5, ax=ax)
    add("R-5 Correlación de Parciales", fig,
        "Todas las correlaciones son superiores a 0.87 lo que significa que hay una gran probabilidad de que mantengas el ritmo a lo largo de la maratón.")

    #  R-6  Edad vs Velocidad 
    fig, ax = plt.subplots(figsize=(10, 6))
    for gen, color, lbl in [('M', 'blue', 'Masculino'), ('F', 'red', 'Femenino')]:
        sub = df[df['M/F'] == gen].sample(
            min(3000, (df['M/F'] == gen).sum()), random_state=42)
        ax.scatter(sub['Age'], sub['velocidad_kmh'],
                   alpha=0.25, s=8, color=color, label=lbl)
        m, b = np.polyfit(sub['Age'], sub['velocidad_kmh'], 1)
        xs = np.linspace(sub['Age'].min(), sub['Age'].max(), 100)
        ax.plot(xs, m * xs + b, color=color, lw=2.5)
    ax.set(xlabel='Edad', ylabel='Velocidad media (km/h)')
    ax.legend(fontsize=11)
    add("R-6 Edad vs Velocidad", fig,
        "La velocidad decrece linealmente con la edad pero se observa como los hombres mantienen ~2 km/h de ventaja en todos los rangos.")

    #  R-7  Ritmo primera vs segunda Mitad 
    diff = df['ritmo_diferencia'].dropna()
    diff = diff[(diff > -5) & (diff < 15)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(diff[diff > 0],  bins=80, color='#E53935', alpha=0.75,
            label=f'Se ralentizó ({(diff > 0).sum():,})')
    ax.hist(diff[diff <= 0], bins=30, color='#43A047', alpha=0.75,
            label=f'Negative split ({(diff <= 0).sum():,})')
    ax.axvline(diff.mean(), color='gold', lw=2, ls='--',
               label=f'Media: +{diff.mean():.2f} min/km')
    ax.set(xlabel='Pace (min/km)', ylabel='Número de corredores')
    ax.legend(fontsize=10)
    add("R-7 Ritmo de Carrera", fig,
        "El 85% pierde ritmo en la 2ª mitad, y hay una diferencia media de ~1.5 min/km, que nos deja intuir donde empieza el 'muro'.")

    #  R-8  Élite vs Amateur 
    fig, ax = plt.subplots(figsize=(10, 5))
    for cat, color, bins in [('Amateur (≥2.5h)', 'grey', 80),
                              ('Élite (<2.5h)',   'orange',   20)]:
        sub = df.loc[df['categoria'] == cat, 'Official_Time_min']
        ax.hist(sub, bins=bins, color=color, alpha=0.75,
                label=f'{cat} ({len(sub):,})')
    ax.axvline(150, color='red', lw=2, ls='--', label='Umbral 2.5 h')
    ax.set(xlabel='Tiempo (min)', ylabel='Número de corredores')
    _fmt_tiempo(ax, 'x')
    ax.legend(fontsize=10)
    pct_elite = (df['categoria'] == 'Élite (<2.5h)').mean() * 100
    add("R-8 Élite vs Amateur", fig,
        f"Solo el {pct_elite:.1f}% completa el maratón en menos de 2 horas y 30 minutos, lo que resulta en una diferencia tan notoria que cuesta distinguir el naranja de los corredores élite.")

    #  R-9  Consistencia de Ritmo por Género 
    fig, ax = plt.subplots(figsize=(8, 6))
    sub_r9 = df[df['ritmo_diferencia'].between(-5, 15)]
    sns.violinplot(data=sub_r9, x='M/F', y='ritmo_diferencia',
                   palette={'M': 'blue', 'F': 'red'}, inner='quart', ax=ax)
    ax.axhline(0, color='black', lw=1.5, ls='--')
    ax.set(xlabel='Género', ylabel='Pace (min/km)')
    add("R-9 Consistencia de Ritmo", fig,
        "En este gráfico vemos como las mujeres muestran una distribución de ritmo ligeramente más constante.")

    #  R-10  Mediana de Tiempo por Edad 
    fig, ax = plt.subplots(figsize=(11, 6))
    for gen, color, lbl in [('M', 'blue', 'Masculino'), ('F', 'red', 'Femenino')]:
        med = (df[df['M/F'] == gen]
               .groupby('Age')['Official_Time_min']
               .median()
               .rolling(3, center=True).mean())
        ax.plot(med.index, med.values, color=color, lw=2.5, label=lbl)
    _fmt_tiempo(ax, 'y')
    ax.set(xlabel='Edad', ylabel='Tiempo')
    ax.legend(fontsize=10)
    add("R-10 Mediana por Edad", fig,
        "El máximo rendimiento lo encontramos entre los 25 y 35 años para ambos géneros.")

    #  R-11  Relación 5K vs 40K 
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(df['5K_h'] * 60, df['40K_h'] * 60,
                    c=df['Official_Time_min'], cmap='plasma', alpha=0.3, s=10)
    plt.colorbar(sc, label='Tiempo final (min)')
    ax.set(xlabel='Minutos al 5K', ylabel='Minutos al 40K')
    add("R-11 5K vs 40K", fig,
        "Aquí observamos una correlación lineal muy alta entre paso inicial y paso en el km 40, significa que quienes salen rápido llegan también rápido al 40K.")

    #  R-12  Origen por Estado (USA) 
    top_st = df[df['Country'] == 'USA']['State'].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    colores_r12 = ['#FF6F00' if s == 'MA' else '#1565C0' for s in top_st.index]
    sns.barplot(x=top_st.index, y=top_st.values, palette=colores_r12, ax=ax)
    ax.set(xlabel='Estado', ylabel='Corredores')
    add("R-12 Origen por Estado", fig,
        "Aquí, debido a la localía ya que la maratón se lleva a cabo en Boston, domina con mucha diferencia Massachusetts.")

    #  R-13  Fatiga Acumulada  
    fig, ax = plt.subplots(figsize=(12, 6))
    for gen, color, lbl in [('M', 'blue', 'Masculino'), ('F', 'red', 'Femenino')]:
        vals = [(df.loc[df['M/F'] == gen, c].mean() * 60) / 5 for c in SEG_COLS]
        ax.plot(ETQ_SEG, vals, marker='o', lw=2.5, color=color, label=lbl)
    ax.axvspan(5.5, 6.5, alpha=0.15, color='red', label='Zona crítica "El Muro"')
    ax.set(xlabel='Segmento', ylabel='Pace medio (min/km)')
    ax.legend(fontsize=10)
    add("R-13 Fatiga por Segmentos", fig,
        "El deterioro de ritmo más pronunciado ocurre en los segmentos 30–35K y 35–40K, coincidiendo con el 'muro' fisiológico.")

    #  R-14  Rangos de Velocidad 
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.histplot(df['velocidad_kmh'], bins=60, color='lightblue', ax=ax)
    ax.axvline(df['velocidad_kmh'].mean(), color='navy', ls='--',
               label=f"Media: {df['velocidad_kmh'].mean():.1f} km/h")
    ax.set(xlabel='km/h', ylabel='Corredores')
    ax.legend(fontsize=10)
    add("R-14 Rangos de Velocidad", fig,
        "La mayoría de los corredores se sitúan en el rango de 10 a 13 km/h, ligeramente a la derecha de la media que esta representada en azul oscuro.")

#  R-15  Categorías Competitivas (Re-hechas con el género y la edad)
    cortes, etiq = [18, 34, 39, 44, 49, 54, 59, 64, 69, 100], ['18-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70+']
    conteos_r15 = (df['M/F'] + " " + pd.cut(df['Age'], bins=cortes, labels=etiq).astype(str)).value_counts().head(15)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = sns.barplot(x=conteos_r15.index, y=conteos_r15.values, palette="viridis", edgecolor='white', ax=ax)   
    ax.set_xticks(range(len(conteos_r15)))
    ax.set_xticklabels(conteos_r15.index, rotation=30, ha='right', fontsize=9)    
    for bar in bars.patches:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50, 
                f'{int(bar.get_height()):,}', ha='center', fontsize=8)
    ax.set(xlabel='Categoría Oficial (Género y Edad)', ylabel='Número de corredores')
    ax.set_ylim(0, conteos_r15.max() * 1.15)
    add("R-15 Categorías Competitivas", fig,
        "Las categorías de edad media (35-49 años) en hombres y mujeres son las que mayor participación concentran, siendo 'M 45-49' y 'M 40-44' las más numerosas.")

    #  R-16  Densidad 10K vs Tiempo Final 
    fig, ax = plt.subplots(figsize=(9, 7))
    hb = ax.hexbin(df['10K_h'] * 60, df['Official_Time_min'],
                   gridsize=40, cmap='hot', mincnt=1)
    _fmt_tiempo(ax, 'y')
    plt.colorbar(hb, label='Densidad')
    ax.set(xlabel='10K (min)', ylabel='Final (min)')
    add("R-16 Densidad 10K vs Final", fig,
        "El gráfico 'hexbin' ayuda a confirmar una fuerte correlación lineal positiva. El núcleo de mayor densidad muestra que el gran pelotón cruza los 10K en ~55 minutos, proyectando un tiempo final en torno a 3h45m - 4h00m.")
    
    #  R-17  Mayores de 60 Años 
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(df[df['Age'] < 60]['Official_Time_min'],
                color='blue', label='< 60 años', fill=True, ax=ax)
    sns.kdeplot(df[df['Age'] >= 60]['Official_Time_min'],
                color='orange', label='≥ 60 años', fill=True, ax=ax)
    _fmt_tiempo(ax, 'x')
    ax.legend(fontsize=10)
    add("R-17 Mayores de 60", fig,
        "El grupo senior muestra en las curvas de densidad una curva desplazada ~60 min a la derecha pero muy estable. Confirmando que el envejecimiento penaliza en el rendimiento final.")

    #  R-18  Velocidad Media por País (Top 5) 
    top5_c = df['Country'].value_counts().head(5).index
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df[df['Country'].isin(top5_c)],
                x='Country', y='velocidad_kmh', order=top5_c, ax=ax)
    ax.set(ylabel='km/h')
    add("R-18 Velocidad por País", fig,
        "Podemos observar como USA tiene la media más baja que podemos atribuir a su altísima participación amateur. El resto de países tienen una mayor media que puede deberse a las marcas que deben superar sus corredores para viajar y presentarse a la maratón.")

    #  R-19  Identificación de Outliers 
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.boxplot(x=df['Official_Time_min'], color='red',
                flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.2}, ax=ax)
    _fmt_tiempo(ax, 'x')
    add("R-19 Outliers", fig,
        "Llaman la atención los valores extremos por encima de las 6 horas, que han podido ser errores de registro o corredores que terminaron andando o sufrieron alguna lesión. Sin embargo la ausencia de valores extremos en la zona inferior dejan claro el límite humano.")

    #  R-20  Distribución del Pace (seg/km) 
    pace = df['Pace_seg_km'].dropna()
    pace = pace[(pace > 150) & (pace < 600)]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(pace, bins=60, color='#7B1FA2', alpha=0.75, kde=True,
                 line_kws={'color': '#4A148C', 'lw': 2}, ax=ax)
    ax.axvline(pace.mean(), color='gold', lw=2, ls='--',
               label=f'Media: {pace.mean():.0f} seg/km')
    ax.set(xlabel='Pace (seg/km)', ylabel='Número de corredores')
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x // 60)}:{int(x % 60):02d}/km"))
    ax.legend(fontsize=10)
    add("R-20 Pace por Kilómetro", fig,
        "En esta distribucion encontramos un pico de máxima frecuencia agrupando a los corredores que logran un ritmo de entre 5:00 y 6:00 min/km.")

  #  HALLAZGOS 
    
    #  H01  Negative Splits 
    neg = df[df['ritmo_diferencia'] < 0]
    pos = df[df['ritmo_diferencia'] >= 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    _, _, textos_h1 = axes[0].pie(
        [len(neg), len(pos)],
        labels=[f'Neg. split\n({len(neg):,})', f'Pos. split\n({len(pos):,})'],
        colors=['#43A047', '#E53935'],
        autopct='%1.1f%%', pctdistance=0.82,
        wedgeprops=dict(width=0.6, edgecolor='white')
    )
    plt.setp(textos_h1, color='white', weight='bold')

    df_split = df.assign(tipo=np.where(df['ritmo_diferencia'] < 0, 'Negative', 'Positive'))
    sns.boxplot(data=df_split, x='tipo', y='Official_Time_min',
                palette={'Negative': '#43A047', 'Positive': '#E53935'}, ax=axes[1])
    axes[1].set(title='Tiempo final comparado', xlabel='', ylabel='Tiempo')
    _fmt_tiempo(axes[1], 'y')

    add("H01 Negative Splits", fig,
        f"Solo el 3,1% logra correr la segunda mitad más rápido, mejorando su tiempo final unos 10 minutos de media."
        f" Pero llama más todavía la atención la varianza, el grupo de negative split presenta un rango más estrecho,"
        f" mientras que el otro grupo arrastra a los outliers que superan las 5h30m."
        f" También llaman la atención los perfiles extremos de {df['Age'].min()} y {df['Age'].max()} años.")

    #  H02  El Muro — Degradación Relativa del Ritmo 
    fig, ax = plt.subplots(figsize=(12, 6))
    for gen, color, lbl in [('M', 'blue', 'Masculino'), ('F', 'red', 'Femenino')]:
        pace_abs  = np.array([(df.loc[df['M/F'] == gen, c].mean() * 60) / 5
                               for c in SEG_COLS])
        pace_base = pace_abs[0]
        degradacion = ((pace_abs - pace_base) / pace_base) * 100
        ax.plot(ETQ_SEG, degradacion, marker='o', lw=2.5, color=color, label=lbl)

    ax.axhspan(0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 30,
               alpha=0.04, color='red')
    ax.axvspan(5.5, 6.5, alpha=0.2, color='red', label='Zona "El Muro" (30–35K)')
    ax.axhline(0, color='gray', ls=':', lw=1, label='Sin degradación')
    ax.set(title='Degradación Relativa del Ritmo por Segmento (%)',
           xlabel='Segmento', ylabel='Incremento del pace respecto al 1er segmento (%)')
    ax.legend(fontsize=10)
    add("H02 El Muro", fig,
        "Ahondando en la investigación del muro encontramos que el deterioro máximo ocurre entre los km 30 y 35 aproximadamente un +8–10% mas que el ritmo inicial."
        " Lo que más llama la atención es que mientras los hombres estabilizan un poco la caída, las mujeres presentan mayor fatiga, lo que significa una posible mayor caía al 'encontrarse con el muro'")

    #  H03  ¿Inicio Rápido Causa Peor Segunda Mitad? 

    sub = df.dropna(subset=['seg_0_5K', 'ritmo_diferencia']).copy()
    sub['decil'] = pd.cut(
        100 - sub['seg_0_5K'].rank(pct=True) * 100,
        bins=10,
        labels=[f'{i * 10}–{(i + 1) * 10}%' for i in range(10)]
    )
    media_deg = sub.groupby('decil', observed=True)['ritmo_diferencia'].mean()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(media_deg)), media_deg.values,
           color=['#43A047' if v < 0 else '#E53935' for v in media_deg],
           edgecolor='white')
    ax.set_xticks(range(len(media_deg)))
    ax.set_xticklabels(media_deg.index, rotation=30, ha='right', fontsize=9)
    ax.axhline(0, color='black', lw=1.5)
    ax.set(xlabel='Decil de velocidad en los primeros 5K',
           ylabel='Degradación media del pace en la 2ª mitad (min/km)')
    add("H03 Inicio Rápido", fig,
        "Con firmamos como una salida rápida en los primeros genera más de 2 min/km adicionales en la segunda mitad. Los corredores con un inicio más conservador son los únicos capaces de mantener o mejorar el ritmo."
        )

    #  H04  Negative Splits por Grupo de Edad 
    df2 = df.dropna(subset=['Age', 'ritmo_diferencia']).copy()
    df2['grupo'] = pd.cut(df2['Age'],
                          bins=[18, 25, 35, 45, 55, 65, 100],
                          labels=['18–25', '26–35', '36–45', '46–55', '56–65', '66+'])
    pct_neg = (df2.groupby('grupo', observed=True)
               .apply(lambda x: (x['ritmo_diferencia'] < 0).mean() * 100)
               .reset_index(name='pct'))
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=pct_neg, x='grupo', y='pct',
                palette=sns.color_palette("viridis", len(pct_neg)), ax=ax)
    for bar, val in zip(ax.patches, pct_neg['pct']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f'{val:.1f}%', ha='center', fontsize=10)
    ax.set(xlabel='Grupo de edad', ylabel='% con negative split',
           ylim=(0, pct_neg['pct'].max() * 1.2))
    add("H04 Edad y Negative Splits", fig,
        "Podemos ver como cuanto mayor es el corredor, menos probable se vuelve que consiga un mejor tiempo en la segunda parte respecto a la primera."
        )

    #  H05  EE.UU. vs Internacionales 
    usa  = df.loc[df['Country'] == 'USA', 'Official_Time_min']
    intl = df.loc[df['Country'] != 'USA', 'Official_Time_min']
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(usa,  bins=80, density=True, alpha=0.55, color='blue',
            label=f'EE.UU. ({len(usa):,})')
    ax.hist(intl, bins=80, density=True, alpha=0.75, color='orange',
            label=f'Internacional ({len(intl):,})')
    ax.axvline(usa.median(),  color='blue',     lw=2.5, ls='--',
               label=f'Mediana USA: {horas_a_hhmmss(usa.median() / 60)}')
    ax.axvline(intl.median(), color='orange', lw=2.5, ls='--',
               label=f'Mediana Intl: {horas_a_hhmmss(intl.median() / 60)}')
    ax.set(xlabel='Tiempo final (min)', ylabel='Densidad')
    _fmt_tiempo(ax, 'x')
    ax.legend(fontsize=10)
    add("H05 Internacionales vs EE.UU.", fig,
        "Ambas poblaciones presentan distribuciones asimétricas positivas (con la mayor concentración a la izquierda de la media y mediana)."
        " Sin embargo, la población internacional no solo desplaza su media aproximadamente 20 minutos, sino que tambien presenta menor desviación estándar."
)

    return vis

