"""
Simulador de Ventas - Noviembre 2025
=====================================
App de Streamlit para simular y visualizar predicciones de ventas usando
el modelo HistGradientBoostingRegressor entrenado (models/modelo_final_true.joblib).

IMPORTANTE - features reales del modelo (versión con Black Friday/Cyber Monday/festivo,
la de mejor performance validada: MAE 1.55 / R² 0.771 sobre 2024):
['precio_base', 'es_estrella', 'precio_venta', 'Amazon', 'Decathlon', 'Deporvillage',
 'Año', 'dif_amazon_pct', 'dif_decathlon_pct', 'dif_deporvillage_pct',
 'es_black_friday', 'es_cyber_monday', 'es_festivo']

El modelo NO usa lags ni media móvil (esas columnas se crearon solo para
df_interferencia en el notebook, pero el modelo se entrenó ANTES de que
existieran). Por eso esta app predice los 30 días de forma vectorizada,
sin recursividad de lags.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Simulador de Ventas - Noviembre 2025",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "models/modelo_final_true.joblib"
DATA_PATH = "data/processed/inferencia_df_transformado.true.csv"

FEATURE_COLS = [
    "precio_base", "es_estrella", "precio_venta", "Amazon", "Decathlon",
    "Deporvillage", "Año", "dif_amazon_pct", "dif_decathlon_pct",
    "dif_deporvillage_pct", "es_black_friday", "es_cyber_monday", "es_festivo",
]

COLS_BOOLEANAS = ["es_estrella", "es_black_friday", "es_cyber_monday", "es_festivo"]

COMPETIDORES = ["Amazon", "Decathlon", "Deporvillage"]
DIF_COLS = {"Amazon": "dif_amazon_pct", "Decathlon": "dif_decathlon_pct",
            "Deporvillage": "dif_deporvillage_pct"}

DIAS_ES = {
    "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
    "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado",
    "Sunday": "Domingo",
}

ESCENARIOS = {
    "Actual (0%)": 1.00,
    "Competencia -5%": 0.95,
    "Competencia +5%": 1.05,
}

# ----------------------------------------------------------------------------
# ESTILOS (paleta morada/azul)
# ----------------------------------------------------------------------------
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.8rem 2rem;
    border-radius: 14px;
    color: white;
    margin-bottom: 1.5rem;
}
.main-header h1 { margin: 0; font-size: 1.8rem; }
.main-header p { margin: 0.3rem 0 0 0; opacity: 0.9; }
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
    border: 1px solid #ddd6fe;
    border-radius: 12px;
    padding: 1rem;
}
.escenario-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 14px;
    padding: 1.2rem;
    color: white;
    text-align: center;
}
.escenario-card h4 { margin: 0 0 0.6rem 0; }
.escenario-card .valor { font-size: 1.5rem; font-weight: 700; }
hr { margin: 1.6rem 0; }
</style>
""", unsafe_allow_html=True)

sns.set_style("whitegrid")

# ----------------------------------------------------------------------------
# CARGA DE MODELO Y DATOS
# ----------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    return joblib.load(MODEL_PATH)


def _es_black_friday(fecha: pd.Timestamp) -> bool:
    if fecha.month == 11:
        nov = pd.date_range(f"{fecha.year}-11-01", f"{fecha.year}-11-30", freq="D")
        viernes = nov[nov.weekday == 4]
        if len(viernes):
            return fecha == viernes[-1]
    return False


def _es_cyber_monday(fecha: pd.Timestamp) -> bool:
    nov = pd.date_range(f"{fecha.year}-11-01", f"{fecha.year}-11-30", freq="D")
    viernes = nov[nov.weekday == 4]
    if len(viernes):
        cyber_monday = viernes[-1] + pd.Timedelta(days=3)
        return fecha == cyber_monday
    return False


@st.cache_data
def cargar_datos():
    df = pd.read_csv(DATA_PATH)
    df["fecha"] = pd.to_datetime(df["fecha"])

    if "Año" not in df.columns and "año" in df.columns:
        df["Año"] = df["año"]

    # Respaldo: si el CSV no trajera estas columnas, se calculan acá desde 'fecha'
    if "es_black_friday" not in df.columns:
        df["es_black_friday"] = df["fecha"].apply(_es_black_friday)
    if "es_cyber_monday" not in df.columns:
        df["es_cyber_monday"] = df["fecha"].apply(_es_cyber_monday)
    if "es_festivo" not in df.columns:
        try:
            import holidays
            es_holidays = holidays.Spain()
            df["es_festivo"] = df["fecha"].apply(lambda x: x in es_holidays)
        except ImportError:
            df["es_festivo"] = False

    for col in COLS_BOOLEANAS:
        df[col] = df[col].astype(int)

    df["dia_semana"] = df["fecha"].dt.day_name().map(DIAS_ES)
    return df


try:
    modelo = cargar_modelo()
except Exception as e:
    st.error(f"⚠️ No se pudo cargar el modelo desde `{MODEL_PATH}`. Detalle: {e}")
    st.stop()

try:
    df_base = cargar_datos()
except Exception as e:
    st.error(f"⚠️ No se pudo cargar el dataset desde `{DATA_PATH}`. Detalle: {e}")
    st.stop()

faltantes = [c for c in FEATURE_COLS if c not in df_base.columns]
if faltantes:
    st.error(f"⚠️ Al dataset le faltan columnas que el modelo necesita: {faltantes}")
    st.stop()


def black_friday_de(anio: int) -> pd.Timestamp:
    nov = pd.date_range(f"{anio}-11-01", f"{anio}-11-30", freq="D")
    viernes = nov[nov.weekday == 4]
    return viernes[-1]


ANIO = int(df_base["fecha"].dt.year.mode()[0])
BLACK_FRIDAY = black_friday_de(ANIO)

# ----------------------------------------------------------------------------
# LÓGICA DE SIMULACIÓN
# ----------------------------------------------------------------------------
def aplicar_escenario(data: pd.DataFrame, ajuste_descuento_pct: float, factor_competencia: float) -> pd.DataFrame:
    """Recalcula precio_venta y precios/diferenciales de competencia según los controles."""
    d = data.copy()

    # Descuento efectivo actual implícito en los datos, más el ajuste del usuario
    descuento_actual = 1 - (d["precio_venta"] / d["precio_base"])
    nuevo_descuento = (descuento_actual + ajuste_descuento_pct / 100).clip(-0.5, 0.90)
    d["precio_venta"] = d["precio_base"] * (1 - nuevo_descuento)

    # Escenario de competencia
    for comp in COMPETIDORES:
        d[comp] = d[comp] * factor_competencia

    # NOTA: asumimos dif_x_pct = (precio_venta - competidor) / competidor * 100.
    # Verificá esta fórmula contra un dato real tuyo (la original viene de tus CSV fuente,
    # no del notebook de entrenamiento) y ajustala acá si difiere.
    for comp, col in DIF_COLS.items():
        d[col] = (d["precio_venta"] - d[comp]) / d[comp] * 100

    return d


def predecir(data: pd.DataFrame) -> np.ndarray:
    X = data[FEATURE_COLS].copy()
    preds = modelo.predict(X)
    return np.clip(preds, 0, None)


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
st.sidebar.title("🎛️ Controles de Simulación")

productos = sorted(df_base["nombre"].unique())
producto_sel = st.sidebar.selectbox("📦 Producto", productos)

ajuste_descuento = st.sidebar.slider(
    "💸 Ajuste de descuento", min_value=-50, max_value=50, value=0, step=5,
    format="%d%%",
)

escenario_sel = st.sidebar.radio(
    "🏷️ Escenario de competencia",
    list(ESCENARIOS.keys()),
)

st.sidebar.markdown("---")
simular = st.sidebar.button("🚀 Simular Ventas", use_container_width=True, type="primary")

if simular:
    st.session_state["simulado"] = True
    st.session_state["producto"] = producto_sel
    st.session_state["descuento"] = ajuste_descuento
    st.session_state["escenario"] = escenario_sel

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown(f"""
<div class="main-header">
    <h1>🛒 Dashboard de Simulación de Ventas — Noviembre {ANIO}</h1>
    <p>Producto seleccionado: <b>{producto_sel}</b></p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.get("simulado"):
    st.info("👈 Ajustá los controles en el sidebar y presioná **Simular Ventas** para ver los resultados.")
    st.stop()

with st.spinner("Calculando predicciones..."):
    producto = st.session_state["producto"]
    descuento = st.session_state["descuento"]
    escenario = st.session_state["escenario"]

    df_producto = df_base[df_base["nombre"] == producto].sort_values("fecha").reset_index(drop=True)

    if df_producto.empty:
        st.warning("No hay datos de noviembre para este producto.")
        st.stop()

    factor = ESCENARIOS[escenario]
    df_sim = aplicar_escenario(df_producto, descuento, factor)
    df_sim["unidades_predichas"] = predecir(df_sim)
    df_sim["ingresos_predichos"] = df_sim["unidades_predichas"] * df_sim["precio_venta"]
    df_sim["descuento_aplicado_pct"] = (1 - df_sim["precio_venta"] / df_sim["precio_base"]) * 100
    df_sim["precio_competencia_prom"] = df_sim[COMPETIDORES].mean(axis=1)
    df_sim["es_black_friday"] = df_sim["fecha"] == BLACK_FRIDAY

# ----------------------------------------------------------------------------
# KPIs
# ----------------------------------------------------------------------------
unidades_totales = df_sim["unidades_predichas"].sum()
ingresos_totales = df_sim["ingresos_predichos"].sum()
precio_prom = df_sim["precio_venta"].mean()
descuento_prom = df_sim["descuento_aplicado_pct"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("📦 Unidades totales proyectadas", f"{unidades_totales:,.0f}".replace(",", "."))
c2.metric("💰 Ingresos proyectados", f"€ {ingresos_totales:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
c3.metric("🏷️ Precio promedio de venta", f"€ {precio_prom:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
c4.metric("🔖 Descuento promedio", f"{descuento_prom:.1f}%")

st.markdown("<hr/>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# GRÁFICO DE PREDICCIÓN DIARIA
# ----------------------------------------------------------------------------
st.subheader("📈 Predicción diaria de ventas — Noviembre")

fig, ax = plt.subplots(figsize=(11, 4.5))
sns.lineplot(
    data=df_sim, x=df_sim["fecha"].dt.day, y="unidades_predichas",
    ax=ax, color="#764ba2", linewidth=2.4, marker="o", markersize=4,
)

bf_row = df_sim[df_sim["es_black_friday"]]
if not bf_row.empty:
    bf_dia = bf_row["fecha"].dt.day.values[0]
    bf_valor = bf_row["unidades_predichas"].values[0]
    ax.axvline(bf_dia, color="#e63946", linestyle="--", linewidth=1.6, alpha=0.7)
    ax.scatter([bf_dia], [bf_valor], color="#e63946", s=110, zorder=5, edgecolor="white", linewidth=1.5)
    ax.annotate(
        f"🔥 Black Friday\n{bf_valor:.0f} unidades",
        xy=(bf_dia, bf_valor), xytext=(bf_dia - 6, bf_valor * 1.12 if bf_valor > 0 else 1),
        fontsize=9, fontweight="bold", color="#e63946",
        arrowprops=dict(arrowstyle="->", color="#e63946"),
    )

ax.set_xlabel("Día de noviembre")
ax.set_ylabel("Unidades vendidas (predicción)")
ax.set_xticks(range(1, 31))
sns.despine(ax=ax)
fig.tight_layout()
st.pyplot(fig)

st.markdown("<hr/>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# TABLA DETALLADA
# ----------------------------------------------------------------------------
st.subheader("📋 Detalle diario")

tabla = df_sim[[
    "fecha", "dia_semana", "precio_venta", "precio_competencia_prom",
    "descuento_aplicado_pct", "unidades_predichas", "ingresos_predichos", "es_black_friday",
]].copy()

tabla["fecha"] = tabla["fecha"].dt.strftime("%d/%m/%Y")
tabla["precio_venta"] = tabla["precio_venta"].round(2)
tabla["precio_competencia_prom"] = tabla["precio_competencia_prom"].round(2)
tabla["descuento_aplicado_pct"] = tabla["descuento_aplicado_pct"].round(1)
tabla["unidades_predichas"] = tabla["unidades_predichas"].round(0)
tabla["ingresos_predichos"] = tabla["ingresos_predichos"].round(2)

tabla = tabla.rename(columns={
    "fecha": "Fecha", "dia_semana": "Día", "precio_venta": "Precio venta (€)",
    "precio_competencia_prom": "Precio competencia (€)", "descuento_aplicado_pct": "Descuento (%)",
    "unidades_predichas": "Unidades predichas", "ingresos_predichos": "Ingresos (€)",
})


def resaltar_black_friday(fila):
    if fila["es_black_friday"]:
        return ["background-color: #ffe0e0; font-weight: bold;"] * len(fila)
    return [""] * len(fila)


tabla_mostrar = tabla.drop(columns=["es_black_friday"])
tabla_style = tabla.style.apply(resaltar_black_friday, axis=1).hide(axis="columns", subset=["es_black_friday"])

st.dataframe(tabla_style, use_container_width=True, height=420)
st.caption("🔥 La fila resaltada corresponde al Black Friday (28 de noviembre).")

st.markdown("<hr/>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# COMPARATIVA DE ESCENARIOS DE COMPETENCIA
# ----------------------------------------------------------------------------
st.subheader("⚖️ Comparativa de escenarios de competencia")
st.caption("Se mantiene el descuento elegido en el sidebar y solo varía el precio de la competencia.")

cols = st.columns(3)
for col, (nombre_esc, fact) in zip(cols, ESCENARIOS.items()):
    df_esc = aplicar_escenario(df_producto, descuento, fact)
    df_esc["unidades_predichas"] = predecir(df_esc)
    df_esc["ingresos_predichos"] = df_esc["unidades_predichas"] * df_esc["precio_venta"]

    unidades_esc = df_esc["unidades_predichas"].sum()
    ingresos_esc = df_esc["ingresos_predichos"].sum()

    activo = " ⭐" if nombre_esc == escenario else ""
    with col:
        st.markdown(f"""
        <div class="escenario-card">
            <h4>{nombre_esc}{activo}</h4>
            <div class="valor">{unidades_esc:,.0f}</div>
            <div>unidades totales</div>
            <br/>
            <div class="valor">€ {ingresos_esc:,.0f}</div>
            <div>ingresos proyectados</div>
        </div>
        """.replace(",", "."), unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)
st.caption(
    "⚠️ Nota técnica: el modelo entrenado no utiliza lags ni media móvil de unidades vendidas, "
    "por lo que las predicciones se calculan directamente para cada día de noviembre a partir de "
    "precios, competencia y señales de fecha especial (Black Friday, Cyber Monday, festivos)."
)
