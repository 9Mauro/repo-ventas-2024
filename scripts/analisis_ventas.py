"""
analisis_ventas.py
Escenario B - Análisis de Ventas de una Pequeña Empresa
TP: Gestión Colaborativa, Control de Versiones y Organización Empresarial

Cátedra: Organización Empresarial - UTN TUP
Autor del script: Mauro Villanueva (P2 - Desarrollador Técnico)
Issue de referencia: VENTAS-2

Descripción:
    Procesa el dataset de ventas ubicado en ../datos/ventas.csv y genera
    indicadores clave de desempeño comercial, exportando gráficos y un
    resumen numérico a la carpeta ../resultados/.

    Diseñado para ejecutarse sin modificaciones en Google Colab utilizando
    rutas relativas al archivo de script (compatibilidad garantizada).
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # Backend no interactivo: necesario en Colab/servidor
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Utilidad: limpieza de pantalla multiplataforma
# ---------------------------------------------------------------------------
def limpiar_pantalla() -> None:
    """Limpia la terminal según el sistema operativo en uso."""
    os.system("cls" if os.name == "nt" else "clear")


# ---------------------------------------------------------------------------
# Resolución de rutas relativas al directorio del script
# (funciona igual en ejecución local y en Google Colab)
# ---------------------------------------------------------------------------
DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__))
DIR_BASE   = os.path.join(DIR_SCRIPT, "..")
DIR_DATOS  = os.path.join(DIR_BASE, "datos")
DIR_RES    = os.path.join(DIR_BASE, "resultados")

# Garantizar existencia de la carpeta de resultados
os.makedirs(DIR_RES, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. CARGA Y VALIDACIÓN DEL DATASET
# ---------------------------------------------------------------------------
def cargar_datos(nombre_archivo: str = "ventas.csv") -> pd.DataFrame:
    """
    Carga el archivo CSV y realiza validaciones básicas de integridad.
    Se espera que el dataset tenga las columnas:
        id, producto, cantidad, precio_unitario, fecha_venta
    """
    ruta = os.path.join(DIR_DATOS, nombre_archivo)

    if not os.path.exists(ruta):
        # Abortamos con un mensaje claro en lugar de un traceback críptico
        print(f"\n[ERROR] No se encontró el archivo: {ruta}")
        print("  Verificar que el dataset esté en la carpeta /datos/\n")
        sys.exit(1)

    df = pd.read_csv(ruta, parse_dates=["fecha_venta"])

    # Verificar columnas requeridas
    columnas_req = {"id", "producto", "cantidad", "precio_unitario", "fecha_venta"}
    faltantes = columnas_req - set(df.columns)
    if faltantes:
        print(f"\n[ERROR] Columnas faltantes en el dataset: {faltantes}\n")
        sys.exit(1)

    # Columna derivada: importe total por fila (cantidad × precio)
    # Permite calcular ingresos reales, no solo unidades
    df["importe_venta"] = df["cantidad"] * df["precio_unitario"]

    # Extraer mes y año para agrupaciones temporales
    df["mes"]      = df["fecha_venta"].dt.month
    df["anio"]     = df["fecha_venta"].dt.year
    df["mes_anio"] = df["fecha_venta"].dt.to_period("M")

    return df


# ---------------------------------------------------------------------------
# 2. INDICADORES CLAVE (KPIs)
# ---------------------------------------------------------------------------
def calcular_indicadores(df: pd.DataFrame) -> dict:
    """
    Calcula los indicadores comerciales solicitados en el escenario:
        - Ventas totales (importe acumulado)
        - Producto más vendido (por cantidad de unidades)
        - Ventas por mes (importe agregado mensualmente)
    Retorna un diccionario con cada KPI para facilitar el reporte.
    """
    total_ingresos = df["importe_venta"].sum()

    # Producto más vendido: suma de cantidades por producto
    unidades_por_producto = df.groupby("producto")["cantidad"].sum()
    producto_top          = unidades_por_producto.idxmax()
    unidades_top          = unidades_por_producto.max()

    # Ingresos mensuales ordenados cronológicamente
    ventas_mes = (
        df.groupby("mes_anio")["importe_venta"]
        .sum()
        .sort_index()
    )

    # Importe promedio por transacción
    ticket_promedio = df["importe_venta"].mean()

    return {
        "total_ingresos":       total_ingresos,
        "producto_top":         producto_top,
        "unidades_top":         unidades_top,
        "ventas_mes":           ventas_mes,
        "ticket_promedio":      ticket_promedio,
        "unidades_por_producto": unidades_por_producto.sort_values(ascending=False),
    }


# ---------------------------------------------------------------------------
# 3. GENERACIÓN DE GRÁFICOS
# ---------------------------------------------------------------------------
COLORES_MARCA = ["#2E75B6", "#ED7D31", "#A9D18E", "#FF0000", "#7030A0"]


def grafico_evolucion_mensual(ventas_mes: pd.Series) -> str:
    """
    Genera gráfico de línea: evolución del importe de ventas mes a mes.
    Exporta PNG a /resultados/ y retorna la ruta del archivo.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Convertir Period a string para el eje X (evita problemas en Colab)
    etiquetas = [str(p) for p in ventas_mes.index]
    valores   = ventas_mes.values

    ax.plot(etiquetas, valores, marker="o", linewidth=2,
            color=COLORES_MARCA[0], markersize=6)
    ax.fill_between(etiquetas, valores, alpha=0.12, color=COLORES_MARCA[0])

    # Anotación del pico máximo
    idx_max = valores.argmax()
    ax.annotate(
        f"Pico: ${valores[idx_max]:,.0f}",
        xy=(etiquetas[idx_max], valores[idx_max]),
        xytext=(0, 12), textcoords="offset points",
        ha="center", fontsize=8, color=COLORES_MARCA[0]
    )

    ax.set_title("Evolución mensual de ingresos por ventas (2024)", fontsize=13, pad=12)
    ax.set_xlabel("Mes")
    ax.set_ylabel("Importe ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()

    ruta = os.path.join(DIR_RES, "evolucion_ventas_mensual.png")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def grafico_ventas_por_producto(unidades_por_producto: pd.Series) -> str:
    """
    Genera gráfico de barras horizontales: unidades vendidas por producto.
    Permite identificar visualmente el producto de mayor rotación.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    colores = COLORES_MARCA[:len(unidades_por_producto)]
    ax.barh(
        unidades_por_producto.index,
        unidades_por_producto.values,
        color=colores,
        edgecolor="white"
    )

    # Etiquetas de valor al final de cada barra
    for i, v in enumerate(unidades_por_producto.values):
        ax.text(v + 1, i, str(int(v)), va="center", fontsize=9)

    ax.set_title("Unidades vendidas por producto (acumulado 2024)", fontsize=13, pad=12)
    ax.set_xlabel("Unidades vendidas")
    ax.invert_yaxis()   # El más vendido queda arriba
    plt.tight_layout()

    ruta = os.path.join(DIR_RES, "ventas_por_producto.png")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


# ---------------------------------------------------------------------------
# 4. EXPORTACIÓN DE RESUMEN NUMÉRICO
# ---------------------------------------------------------------------------
def exportar_resumen(df: pd.DataFrame, kpis: dict) -> str:
    """
    Guarda un archivo CSV con el resumen de ventas mensuales.
    Facilita la revisión de resultados sin necesidad de re-ejecutar el script.
    """
    resumen = kpis["ventas_mes"].reset_index()
    resumen.columns = ["mes", "importe_total"]
    resumen["importe_total"] = resumen["importe_total"].round(2)

    ruta = os.path.join(DIR_RES, "resumen_ventas_mensual.csv")
    resumen.to_csv(ruta, index=False)
    return ruta


# ---------------------------------------------------------------------------
# 5. REPORTE EN CONSOLA
# ---------------------------------------------------------------------------
def imprimir_reporte(kpis: dict) -> None:
    """Muestra los indicadores principales formateados en la terminal."""
    sep = "-" * 52

    print("\n" + sep)
    print("  INFORME DE VENTAS - UTN TUP - ORGANIZACIÓN EMPRESARIAL")
    print(sep)

    print(f"\n  Total de ingresos (2024)  : $ {kpis['total_ingresos']:>14,.2f}")
    print(f"  Ticket promedio por venta : $ {kpis['ticket_promedio']:>14,.2f}")
    print(f"  Producto más vendido      : {kpis['producto_top']} ({int(kpis['unidades_top'])} unidades)")

    print("\n  Ingresos por mes:")
    print(f"  {'Mes':<12}  {'Importe':>14}")
    print("  " + "-" * 30)
    for periodo, importe in kpis["ventas_mes"].items():
        print(f"  {str(periodo):<12}  $ {importe:>12,.2f}")

    print("\n" + sep + "\n")


# ---------------------------------------------------------------------------
# PUNTO DE ENTRADA
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    limpiar_pantalla()

    print("\nCargando datos de ventas...")
    df = cargar_datos()

    print("Calculando indicadores...")
    kpis = calcular_indicadores(df)

    print("Generando gráficos...")
    ruta_g1 = grafico_evolucion_mensual(kpis["ventas_mes"])
    ruta_g2 = grafico_ventas_por_producto(kpis["unidades_por_producto"])

    print("Exportando resumen CSV...")
    ruta_csv = exportar_resumen(df, kpis)

    imprimir_reporte(kpis)

    print(f"  Gráfico 1 guardado en: {ruta_g1}")
    print(f"  Gráfico 2 guardado en: {ruta_g2}")
    print(f"  Resumen  guardado en: {ruta_csv}\n")
