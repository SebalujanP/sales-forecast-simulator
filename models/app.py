import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Análisis de Ventas",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Análisis de Ventas de Tienda Online")
st.markdown("Exploración interactiva de más de **10.000 registros de ventas** para identificar oportunidades de mejora en rentabilidad.")
st.markdown("---")

@st.cache_data
def cargar_datos():
    df = pd.read_csv("samplesuperstore.csv")
    df = df.rename(columns={
        "Row ID": "ID_Fila", "Order ID": "ID_Pedido",
        "Order Date": "Fecha_Pedido", "Ship Date": "Fecha_Envio",
        "Ship Mode": "Modo_Envio", "Customer ID": "ID_Cliente",
        "Customer Name": "Nombre_Cliente", "Segment": "Segmento",
        "Country/Region": "Pais", "City": "Ciudad",
        "State/Province": "Provincia", "Postal Code": "Codigo_Postal",
        "Region": "Region", "Product ID": "ID_Producto",
        "Category": "Categoria", "Sub-Category": "Subcategoria",
        "Product Name": "Nombre_Producto", "Sales": "Ventas",
        "Quantity": "Cantidad", "Discount": "Descuento", "Profit": "Ganancia"
    })
    df['Fecha_Pedido'] = pd.to_datetime(df['Fecha_Pedido'])
    df['Fecha_Envio'] = pd.to_datetime(df['Fecha_Envio'])
    df['Año'] = df['Fecha_Pedido'].dt.year
    df['Mes'] = df['Fecha_Pedido'].dt.month
    return df

df = cargar_datos()

# Sidebar filtros
st.sidebar.header("🔍 Filtros")
categorias = ["Todas"] + list(df['Categoria'].unique())
categoria_sel = st.sidebar.selectbox("Categoría", categorias)

regiones = ["Todas"] + list(df['Region'].unique())
region_sel = st.sidebar.selectbox("Región", regiones)

años = ["Todos"] + sorted(df['Año'].unique().tolist())
año_sel = st.sidebar.selectbox("Año", años)

# Aplicar filtros
df_filtrado = df.copy()
if categoria_sel != "Todas":
    df_filtrado = df_filtrado[df_filtrado['Categoria'] == categoria_sel]
if region_sel != "Todas":
    df_filtrado = df_filtrado[df_filtrado['Region'] == region_sel]
if año_sel != "Todos":
    df_filtrado = df_filtrado[df_filtrado['Año'] == año_sel]

# KPIs
st.subheader("📈 KPIs Generales")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Ventas Totales", f"${df_filtrado['Ventas'].sum():,.0f}")
col2.metric("Ganancia Total", f"${df_filtrado['Ganancia'].sum():,.0f}")
col3.metric("Pedidos", f"{df_filtrado['ID_Pedido'].count():,}")
col4.metric("Ventas con Pérdida", f"{(df_filtrado['Ganancia'] < 0).sum():,}")

st.markdown("---")

# Graficos fila 1
col1, col2 = st.columns(2)

with col1:
    st.subheader("Ventas por Categoría")
    ventas_cat = df_filtrado.groupby('Categoria')['Ventas'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(ventas_cat.index, ventas_cat.values, color='steelblue')
    ax.set_xlabel("Categoría")
    ax.set_ylabel("Ventas")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Ganancias por Categoría")
    gan_cat = df_filtrado.groupby('Categoria')['Ganancia'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['red' if v < 0 else 'steelblue' for v in gan_cat.values]
    ax.bar(gan_cat.index, gan_cat.values, color=colors)
    ax.set_xlabel("Categoría")
    ax.set_ylabel("Ganancia")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# Graficos fila 2
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 10 Subcategorías por Ventas")
    subcat_v = df_filtrado.groupby('Subcategoria')['Ventas'].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(subcat_v.index, subcat_v.values, color='steelblue')
    ax.set_xlabel("Subcategoría")
    ax.set_ylabel("Ventas")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Ventas por Mes")
    ventas_mes = df_filtrado.groupby('Mes')['Ventas'].sum()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(ventas_mes.index, ventas_mes.values, color='steelblue')
    ax.set_xlabel("Mes")
    ax.set_ylabel("Ventas")
    ax.set_xticks(range(1, 13))
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# Graficos fila 3
col1, col2 = st.columns(2)

with col1:
    st.subheader("Ventas por Región")
    ventas_reg = df_filtrado.groupby('Region')['Ventas'].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(ventas_reg.index, ventas_reg.values, color='steelblue')
    ax.set_xlabel("Región")
    ax.set_ylabel("Ventas")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Impacto de Descuentos en Ganancia")
    desc_gan = df_filtrado.groupby('Descuento')['Ganancia'].sum()
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['red' if v < 0 else 'steelblue' for v in desc_gan.values]
    ax.bar(range(len(desc_gan)), desc_gan.values, width=0.6, color=colors)
    ax.set_xticks(range(len(desc_gan)))
    ax.set_xticklabels([f"{x:.2f}" for x in desc_gan.index], rotation=45)
    ax.set_xlabel("Descuento")
    ax.set_ylabel("Ganancia")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# Conclusiones
st.subheader("💡 Conclusiones y Recomendaciones")
st.markdown("""
- **Descuentos del 30%+ generan pérdidas** — Se recomienda limitar los descuentos máximos al 20%.
- **El 18.6% de las transacciones tuvieron ganancia negativa** — 1.901 ventas con pérdida detectadas.
- **Alto volumen no garantiza rentabilidad** — Algunas subcategorías venden mucho pero aportan poco margen.
- **Revisar condiciones de clientes de alto volumen** — Algunos compran mucho pero generan baja ganancia.
""")

st.markdown("---")
st.caption("Proyecto de análisis de datos | Sebastián Luján | github.com/SebalujanP")
