# 🛒 Sales Forecast Simulator

Simulador interactivo de ventas para noviembre 2025, construido sobre un modelo de Machine Learning entrenado con datos históricos (2021-2024) de una tienda de artículos deportivos. Permite ajustar descuentos y escenarios de competencia en tiempo real y visualizar el impacto proyectado en unidades vendidas e ingresos.

🔗 **App en vivo:** https://sales-forecast-simulator-8cst8e6p6vgwfzvmrywazt.streamlit.app/

## 📋 Descripción

El proyecto simula decisiones comerciales de pricing para 24 productos de una tienda de e-commerce deportivo, prediciendo el impacto de:
- Ajustes de descuento sobre el precio base
- Cambios en el precio de la competencia (Amazon, Decathlon, Deporvillage)

La predicción se calcula día por día para todo noviembre 2025, con foco especial en el pico de ventas de **Black Friday**.

## 🧠 Modelo

**Algoritmo:** `HistGradientBoostingRegressor` (scikit-learn)

**Variables utilizadas (13):**
| Variable | Descripción |
|---|---|
| `precio_base` | Precio de lista del producto |
| `es_estrella` | Si el producto es un best-seller de la tienda |
| `precio_venta` | Precio final de venta (con descuento aplicado) |
| `Amazon`, `Decathlon`, `Deporvillage` | Precios de la competencia |
| `Año` | Año del registro |
| `dif_amazon_pct`, `dif_decathlon_pct`, `dif_deporvillage_pct` | Diferencial porcentual de precio vs. cada competidor |
| `es_black_friday` | Si la fecha es el Black Friday del año |
| `es_cyber_monday` | Si la fecha es el Cyber Monday del año |
| `es_festivo` | Si la fecha es feriado en España |

**Validación (split temporal: entrenamiento 2021-2023, validación 2024):**

| Versión del modelo | MAE | R² |
|---|---|---|
| 10 features (sin señales de fecha especial) | 1.80 | 0.693 |
| 13 features (con mes/trimestre/día del mes) | 1.87 | 0.676 |
| **13 features (con Black Friday/Cyber Monday/festivo)** ✅ | **1.55** | **0.771** |

Agregar las señales de Black Friday/Cyber Monday/festivo mejoró notablemente la performance: sin ellas, el modelo no tenía forma de anticipar los picos de demanda de esas fechas, ya que las variables de precio y competencia por sí solas no alcanzan para explicarlos.

El modelo desplegado (`models/modelo_final.joblib`) se reentrenó con el 100% de los datos históricos disponibles (2021-2024) una vez validada esta configuración.

### Decisiones clave y por qué

- **No se usan lags ni media móvil de unidades vendidas**: si bien se calcularon durante la exploración, agregarlas hubiera requerido predicción recursiva día a día (cada predicción alimentando el lag del día siguiente), sumando complejidad y riesgo de propagación de error sin evidencia de que mejoraran el resultado frente a variables de precio/competencia/fecha.
- **No se usa one-hot encoding de producto/categoría**: con pocos registros por producto, agregar ~50 columnas dummy aumenta el riesgo de sobreajuste sin aportar señal más allá de la que ya capturan `es_estrella` y el precio.

## ⚠️ Limitaciones conocidas

- El modelo se entrenó y validó a nivel diario para 24 productos con ~4 años de historia; la performance puede degradarse ante productos nuevos, cambios estructurales de mercado, o promociones no capturadas por las variables actuales (por ejemplo, campañas puntuales fuera de Black Friday/Cyber Monday).
- La fórmula usada para recalcular `dif_x_pct` en la simulación (`(precio_venta - competidor) / competidor * 100`) fue verificada contra los datos históricos, pero no está documentada como parte del pipeline original de generación de datos.

## 🖥️ App (Streamlit)

**Sidebar:**
- Selector de producto
- Slider de ajuste de descuento (-50% a +50%)
- Escenario de competencia (actual / -5% / +5%)

**Dashboard principal:**
- KPIs: unidades totales, ingresos proyectados, precio promedio, descuento promedio
- Gráfico de predicción diaria con Black Friday destacado
- Tabla detallada día por día
- Comparativa entre los 3 escenarios de competencia

## 📁 Estructura del repositorio

```
sales-forecast-simulator/
├── app.py                                          # App de Streamlit
├── requirements.txt
├── models/
│   └── modelo_final.joblib                         # Modelo entrenado (13 features)
└── data/
    └── processed/
        └── inferencia_df_transformado.csv          # Datos de noviembre 2025 (24 productos x 30 días)
```

## 🚀 Cómo correrlo localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🛠️ Stack

- Python, pandas, NumPy
- scikit-learn (`HistGradientBoostingRegressor`)
- Streamlit, seaborn, matplotlib
- joblib

## 👤 Autor

**Sebastián Luján** — Estudiante de Ciencia de Datos e IA (Instituto Superior N°213, Ensenada)
[GitHub](https://github.com/SebalujanP)
