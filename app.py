import streamlit as st
import pandas as pd
import plotly.express as px
from src.database import DatabaseManager
from src.predictor import SalesPredictor
from datetime import datetime, timedelta
from io import BytesIO

#prueba
# Configuración de la página
st.set_page_config(
    page_title="Amazon Analytics",
    page_icon="📊",
    layout="wide"
)

# Inicialización
db = DatabaseManager()
predictor = SalesPredictor()

# Sidebar para navegación
st.sidebar.title("Amazon Analytics")
page = st.sidebar.selectbox(
    "Selecciona una página",
    ["Dashboard", "Productos", "Ventas", "Predicciones", "Reportes", "Métricas"],
    key="main_navigation"
)

# Funciones auxiliares
def load_dashboard_data():
    """Carga los datos para el dashboard principal"""
    with st.spinner('Cargando datos...'):
        # Productos con stock bajo
        low_stock = pd.read_sql("""
            SELECT id, sku, nombre, stock_actual, stock_minimo
            FROM productos
            WHERE stock_actual <= stock_minimo
        """, db.engine)
        
        # Ventas recientes
        recent_sales = pd.read_sql("""
            SELECT p.nombre, v.cantidad, v.precio_venta, v.fecha_venta
            FROM ventas v
            JOIN productos p ON v.producto_id = p.id
            ORDER BY v.fecha_venta DESC
            LIMIT 10
        """, db.engine)
        
        # Métricas generales
        metrics = pd.read_sql("""
            SELECT 
                COUNT(DISTINCT p.id) as total_productos,
                SUM(p.stock_actual) as stock_total,
                AVG(m.rentabilidad) as rentabilidad_promedio
            FROM productos p
            LEFT JOIN metricas m ON p.id = m.producto_id
        """, db.engine)
        
        return low_stock, recent_sales, metrics

# Páginas
if page == "Dashboard":
    st.title("Dashboard Principal")
    
    # Sistema de Alertas
    alertas = db.get_alerts()
    
    if any(len(alerts) > 0 for alerts in alertas.values()):
        st.subheader("Alertas Activas")
        
        # Alertas críticas
        if alertas['critical']:
            st.error("⚠️ Alertas Críticas")
            for alerta in alertas['critical']:
                with st.expander(f"{alerta['tipo']}: {alerta['producto']}"):
                    st.write(f"Categoría: {alerta['categoria']}")
                    st.write(f"Mensaje: {alerta['mensaje']}")
        
        # Alertas de advertencia
        if alertas['warning']:
            st.warning("⚠️ Advertencias")
            for alerta in alertas['warning']:
                with st.expander(f"{alerta['tipo']}: {alerta['producto']}"):
                    st.write(f"Categoría: {alerta['categoria']}")
                    st.write(f"Mensaje: {alerta['mensaje']}")
        
        # Oportunidades
        if alertas['opportunity']:
            st.info("💡 Oportunidades")
            for alerta in alertas['opportunity']:
                with st.expander(f"{alerta['tipo']}: {alerta['producto']}"):
                    st.write(f"Categoría: {alerta['categoria']}")
                    st.write(f"Mensaje: {alerta['mensaje']}")
    
    # Carga de datos
    low_stock, recent_sales, metrics = load_dashboard_data()
    
    # Métricas principales
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Productos", int(metrics['total_productos'].iloc[0]))
    with col2:
        st.metric("Stock Total", int(metrics['stock_total'].iloc[0]))
    with col3:
        st.metric("Rentabilidad Promedio", f"{metrics['rentabilidad_promedio'].iloc[0]:.2f}%")
    
    # Alertas de stock bajo
    st.subheader("Alertas de Stock Bajo")
    if not low_stock.empty:
        st.dataframe(low_stock)
    else:
        st.info("No hay productos con stock bajo")
    
    # Ventas recientes
    st.subheader("Últimas Ventas")
    if not recent_sales.empty:
        st.dataframe(recent_sales)

elif page == "Productos":
    st.title("Gestión de Productos")
    
    # Lista de productos primero
    st.subheader("Lista de Productos")
    productos = pd.read_sql("SELECT * FROM productos", db.engine)
    st.dataframe(productos)
    
    # Sección de edición
    st.subheader("Editar Producto")
    producto_a_editar = st.selectbox(
        "Selecciona un producto para editar",
        options=productos['id'].tolist(),
        format_func=lambda x: productos[productos['id']==x]['nombre'].iloc[0],
        key="selector_editar_producto"
    )
    
    # Obtener datos del producto seleccionado
    producto_actual = productos[productos['id'] == producto_a_editar].iloc[0]
    
    # Formulario de edición
    with st.form(key=f"form_editar_producto"):
        st.write("Modifica los campos que desees actualizar:")
        nuevo_sku = st.text_input("SKU", value=producto_actual['sku'], key="edit_sku")
        nuevo_nombre = st.text_input("Nombre", value=producto_actual['nombre'], key="edit_nombre")
        nuevo_precio_compra = st.number_input("Precio de Compra", 
                                            value=float(producto_actual['precio_compra']), 
                                            min_value=0.0, 
                                            key="edit_precio_compra")
        nuevo_precio_venta = st.number_input("Precio de Venta", 
                                           value=float(producto_actual['precio_venta']), 
                                           min_value=0.0, 
                                           key="edit_precio_venta")
        nuevo_stock_actual = st.number_input("Stock Actual", 
                                           value=int(producto_actual['stock_actual']), 
                                           min_value=0, 
                                           key="edit_stock_actual")
        nuevo_stock_minimo = st.number_input("Stock Mínimo", 
                                           value=int(producto_actual['stock_minimo']), 
                                           min_value=0, 
                                           key="edit_stock_minimo")
        nueva_categoria = st.text_input("Categoría", 
                                      value=producto_actual['categoria'], 
                                      key="edit_categoria")
        
        if st.form_submit_button("Actualizar Producto"):
            try:
                db.update_product(
                    producto_a_editar,
                    nuevo_sku,
                    nuevo_nombre,
                    nuevo_precio_compra,
                    nuevo_precio_venta,
                    nuevo_stock_actual,
                    nuevo_stock_minimo,
                    nueva_categoria
                )
                st.success("Producto actualizado correctamente")
            except Exception as e:
                st.error(f"Error al actualizar producto: {str(e)}")
    
    # Formulario para añadir nuevo producto
    st.subheader("Añadir Nuevo Producto")
    with st.form(key="form_nuevo_producto"):
        st.subheader("Añadir Nuevo Producto")
        sku = st.text_input("SKU", key="input_sku")
        nombre = st.text_input("Nombre", key="input_nombre")
        precio_compra = st.number_input("Precio de Compra", min_value=0.0, key="input_precio_compra")
        precio_venta = st.number_input("Precio de Venta", min_value=0.0, key="input_precio_venta")
        stock_actual = st.number_input("Stock Actual", min_value=0, key="input_stock_actual")
        stock_minimo = st.number_input("Stock Mínimo", min_value=0, key="input_stock_minimo")
        categoria = st.text_input("Categoría", key="input_categoria")
        
        if st.form_submit_button("Añadir Producto"):
            try:
                db.add_product(sku, nombre, precio_compra, precio_venta, 
                             stock_actual, stock_minimo, categoria)
                st.success("Producto añadido correctamente")
            except Exception as e:
                st.error(f"Error al añadir producto: {str(e)}")

elif page == "Ventas":
    st.title("Registro de Ventas")
    
    try:
        # Historial de ventas primero
        st.subheader("Historial de Ventas")
        ventas = pd.read_sql("""
            SELECT p.nombre, v.cantidad, v.precio_venta, v.fecha_venta
            FROM ventas v
            JOIN productos p ON v.producto_id = p.id
            ORDER BY v.fecha_venta DESC
        """, db.engine)
        st.dataframe(ventas)
        
        # Obtener lista de productos
        productos = pd.read_sql("SELECT id, nombre FROM productos", db.engine)
        
        # Formulario para registrar venta
        st.subheader("Registrar Nueva Venta")
        with st.form(key=f"venta_form_{hash('unico')}"):
            producto_id = st.selectbox(
                "Producto", 
                options=productos['id'].tolist(),
                format_func=lambda x: productos[productos['id']==x]['nombre'].iloc[0],
                key="venta_producto_selector_unico"
            )
            cantidad = st.number_input(
                "Cantidad", 
                min_value=1, 
                value=1, 
                key="cantidad_venta_unico"
            )
            precio_venta = st.number_input(
                "Precio de Venta", 
                min_value=0.0, 
                key="precio_venta_unico"
            )
            
            if st.form_submit_button("Registrar Venta"):
                try:
                    db.register_sale(producto_id, cantidad, precio_venta)
                    st.success("Venta registrada correctamente")
                except Exception as e:
                    st.error(f"Error al registrar venta: {str(e)}")
    
    except Exception as e:
        st.error(f"Error en la página de ventas: {str(e)}")

elif page == "Predicciones":
    st.title("Predicciones de Ventas")
    
    # Selector de producto con key única
    productos = pd.read_sql("SELECT id, nombre FROM productos", db.engine)
    producto_id = st.selectbox(
        "Selecciona un producto", 
        options=productos['id'].tolist(),
        format_func=lambda x: productos[productos['id']==x]['nombre'].iloc[0],
        key="prediccion_producto_selector"
    )
    
    if st.button("Generar Predicción", key="btn_generar_prediccion"):
        try:
            with st.spinner('Generando predicción...'):
                # Obtener datos y generar predicción
                ventas_historicas = db.get_product_sales(producto_id)
                
                if not ventas_historicas.empty:
                    prediccion = predictor.predict_sales(ventas_historicas)
                    metricas = predictor.calculate_metrics(prediccion, ventas_historicas)
                    
                    # Obtener stock actual
                    stock_actual = pd.read_sql(f"""
                        SELECT stock_actual 
                        FROM productos 
                        WHERE id = {producto_id}
                    """, db.engine)
                    
                    # Mostrar gráfico principal
                    st.subheader("Predicción de Ventas - Próximos 3 Meses")
                    fig_prediction = px.line(prediccion, x='ds', y=['yhat', 'yhat_lower', 'yhat_upper'],
                                          labels={'value': 'Unidades', 'ds': 'Fecha'},
                                          title="Predicción de Ventas")
                    fig_prediction.add_scatter(x=ventas_historicas['ds'], 
                                            y=ventas_historicas['y'],
                                            name='Ventas Históricas',
                                            mode='markers')
                    st.plotly_chart(fig_prediction)
                    
                    # Métricas clave
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Tendencia", 
                                 f"{metricas['trend_change']:.1f}%",
                                 delta=metricas['trend_change'])
                    with col2:
                        st.metric("Stock Actual",
                                 f"{stock_actual['stock_actual'].iloc[0]} unidades")
                    with col3:
                        st.metric("Stock Óptimo", 
                                 f"{metricas['stock_recommendations']['optimal_stock']:.0f} unidades")
                    with col4:
                        st.metric("Predicción Mensual Promedio",
                                 f"{metricas['monthly_predictions']['yhat'].mean():.1f} unidades/día")
                    
                    # Predicción mensual
                    st.subheader("Predicción Mensual")
                    monthly_df = metricas['monthly_predictions'].reset_index()
                    monthly_df.columns = ['Mes', 'Predicción', 'Mínimo', 'Máximo']
                    st.dataframe(monthly_df.round(2))
                    
                    # Recomendaciones de stock
                    st.subheader("Recomendaciones de Stock")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"Stock Mínimo: {metricas['stock_recommendations']['min_stock']:.0f} unidades")
                    with col2:
                        st.success(f"Stock Óptimo: {metricas['stock_recommendations']['optimal_stock']:.0f} unidades")
                    with col3:
                        st.warning(f"Stock Máximo: {metricas['stock_recommendations']['max_stock']:.0f} unidades")
                else:
                    st.warning("No hay suficientes datos históricos para este producto")
                    
        except Exception as e:
            st.error(f"Error al generar la predicción: {str(e)}")
            st.info("Sugerencia: Asegúrate de que hay suficientes datos de ventas para este producto.")

elif page == "Métricas":
    st.title("Métricas de Rendimiento")
    
    # Métricas generales
    metricas = pd.read_sql("""
        SELECT 
            p.nombre,
            m.rotacion,
            m.rentabilidad,
            m.tendencia_ventas
        FROM productos p
        JOIN metricas m ON p.id = m.producto_id
    """, db.engine)

elif page == "Reportes":
    st.title("Reportes y Exportación")
    
    # Selector de tipo de reporte
    report_type = st.selectbox(
        "Selecciona el tipo de reporte",
        ["Ventas", "Inventario", "Rendimiento"],
        key="report_type_selector"
    )
    
    if report_type == "Ventas":
        st.subheader("Reporte de Ventas")
        
        # Filtros de fecha
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Fecha inicial", key="sales_start_date")
        with col2:
            end_date = st.date_input("Fecha final", key="sales_end_date")
        
        # Generar reporte
        if st.button("Generar Reporte de Ventas"):
            report_df = db.get_sales_report(start_date, end_date)
            
            # Mostrar resumen
            st.write("Resumen de Ventas:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Ventas", f"${report_df['precio_venta'].sum():,.2f}")
            with col2:
                st.metric("Total Beneficio", f"${report_df['beneficio'].sum():,.2f}")
            with col3:
                st.metric("Margen Promedio", f"{report_df['margen_porcentaje'].mean():.1f}%")
            
            # Mostrar datos detallados
            st.dataframe(report_df)
            
            # Botón de descarga
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                report_df.to_excel(writer, index=False)
            
            st.download_button(
                label="Descargar Excel",
                data=excel_buffer.getvalue(),
                file_name=f"reporte_ventas_{start_date}_{end_date}.xlsx",
                mime="application/vnd.ms-excel"
            )
    
    elif report_type == "Inventario":
        st.subheader("Reporte de Inventario")
        
        # Generar reporte
        report_df = db.get_inventory_report()
        
        # Mostrar resumen
        st.write("Resumen de Inventario:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Valor Total Inventario", f"${report_df['valor_inventario'].sum():,.2f}")
        with col2:
            st.metric("Total Productos", len(report_df))
        with col3:
            st.metric("Ventas Totales", f"{report_df['ventas_totales'].sum():,.0f}")
        
        # Mostrar datos detallados
        st.dataframe(report_df)
        
        # Botón de descarga
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            report_df.to_excel(writer, index=False)
        
        st.download_button(
            label="Descargar Excel",
            data=excel_buffer.getvalue(),
            file_name=f"reporte_inventario_{datetime.now().date()}.xlsx",
            mime="application/vnd.ms-excel"
        )
    
    elif report_type == "Rendimiento":
        st.subheader("Reporte de Rendimiento")
        
        # Selector de periodo
        periodo = st.selectbox(
            "Selecciona el periodo",
            ["day", "week", "month", "quarter", "year"],
            format_func=lambda x: {
                "day": "Diario",
                "week": "Semanal",
                "month": "Mensual",
                "quarter": "Trimestral",
                "year": "Anual"
            }[x],
            key="performance_period_selector"
        )
        
        # Generar reporte
        report_df = db.get_performance_report(periodo)
        
        # Mostrar gráfico de tendencia
        fig = px.line(report_df, 
                     x='periodo', 
                     y=['ingresos_totales', 'beneficio_total'],
                     title="Tendencia de Rendimiento")
        st.plotly_chart(fig)
        
        # Mostrar datos detallados
        st.dataframe(report_df)
        
        # Botón de descarga
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            report_df.to_excel(writer, index=False)
        
        st.download_button(
            label="Descargar Excel",
            data=excel_buffer.getvalue(),
            file_name=f"reporte_rendimiento_{periodo}_{datetime.now().date()}.xlsx",
            mime="application/vnd.ms-excel"
        )