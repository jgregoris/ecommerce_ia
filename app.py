import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import BytesIO
from sqlalchemy import text
from src.database import DatabaseManager
from src.predictor import SalesPredictor
from src.clustering import pagina_clustering
from src.anomaly_detection import pagina_anomalias
from src.visualitations import DashboardVisualizer
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de la página
st.set_page_config(
    page_title="Amazon Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicialización con manejo de errores
@st.cache_resource
def init_components():
    try:
        db = DatabaseManager()
        # Verificar conexión
        with db.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        predictor = SalesPredictor()
        visualizer = DashboardVisualizer()
        return db, predictor, visualizer
    except Exception as e:
        logger.error(f"Error al inicializar componentes: {e}")
        st.error(f"Error al inicializar la aplicación: {str(e)}")
        return None, None, None

db, predictor, visualizer = init_components()

if db is None or predictor is None or visualizer is None:
    st.error("No se pudieron inicializar los componentes necesarios.")
    st.stop()

# Sidebar para navegación
with st.sidebar:
    st.title("Amazon Analytics")
    page = st.selectbox(
        "Navegación",
        ["Dashboard", "Productos", "Ventas", "Predicciones", "Reportes", "Métricas", "Clustering", "Anomalías"]
    )
    
    st.sidebar.info(
        "Esta aplicación proporciona análisis avanzado de inventario y ventas utilizando "
        "técnicas de machine learning e inteligencia artificial."
    )

# Funciones auxiliares
def format_currency(value):
    return f"${value:,.2f}"

def format_number(value):
    return f"{value:,.0f}"

# Manejo de páginas
if page == "Dashboard":
    st.title("Dashboard Principal")
    
    try:
        # Cargar datos del dashboard
        low_stock, recent_sales, metrics = db.load_dashboard_data()
        
        if not metrics.empty:
            # Métricas principales
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Ventas (30 días)", 
                    format_currency(metrics['total_ventas'].iloc[0]),
                    f"{format_number(metrics['unidades_vendidas'].iloc[0])} unidades"
                )
            with col2:
                st.metric(
                    "Productos en Catálogo", 
                    format_number(metrics['total_productos'].iloc[0]),
                    f"{metrics['productos_vendidos'].iloc[0]} vendidos"
                )
            with col3:
                st.metric(
                    "Valor de Inventario", 
                    format_currency(metrics['valor_inventario'].iloc[0])
                )
            with col4:
                st.metric(
                    "Stock Total", 
                    format_number(metrics['stock_total'].iloc[0])
                )

        # Selector de producto para predicciones
        try:
            productos_con_ventas = pd.read_sql(
                text("""
                    SELECT 
                        p.id,
                        p.nombre,
                        p.categoria,
                        COUNT(v.id) as total_ventas
                    FROM productos p
                    LEFT JOIN ventas v ON p.id = v.producto_id
                    GROUP BY p.id, p.nombre, p.categoria
                    HAVING COUNT(v.id) > 0
                    ORDER BY COUNT(v.id) DESC
                """),
                db.engine
            )

            if not productos_con_ventas.empty:
                st.subheader("Predicciones de Ventas")
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    producto_id = st.selectbox(
                        "Seleccionar producto para predicciones",
                        options=productos_con_ventas['id'].tolist(),
                        format_func=lambda x: (
                            f"{productos_con_ventas[productos_con_ventas['id']==x]['nombre'].iloc[0]} - "
                            f"{productos_con_ventas[productos_con_ventas['id']==x]['categoria'].iloc[0]} "
                            f"({productos_con_ventas[productos_con_ventas['id']==x]['total_ventas'].iloc[0]} ventas)"
                        )
                    )
                
                with col2:
                    if st.button("Actualizar Dashboard", key="update_dashboard"):
                        st.rerun()

                # Obtener y visualizar predicciones
                sales_data = db.get_sales_report()
                inventory_data = db.get_inventory_report()
                
                if not sales_data.empty and not inventory_data.empty:
                    historical_data = db.get_product_sales(producto_id)
                    if historical_data is not None and not historical_data.empty:
                        predictions_data = predictor.predict_sales(historical_data)
                        if predictions_data is not None:
                            dashboard_fig = visualizer.create_main_dashboard(
                                sales_data,
                                inventory_data,
                                predictions_data
                            )
                            st.plotly_chart(dashboard_fig, use_container_width=True)
                        else:
                            st.warning("No se pudieron generar predicciones para este producto")
                    else:
                        st.warning("No hay suficientes datos históricos para este producto")

        except Exception as e:
            logger.error(f"Error en predicciones: {e}")
            st.error("Error al cargar predicciones de ventas")

        # Sistema de Alertas
        alertas = db.get_alerts()
        if any(len(alerts) > 0 for alerts in alertas.values()):
            st.subheader("Alertas Activas")
            
            col1, col2 = st.columns(2)
            with col1:
                if alertas['critical']:
                    with st.expander("🚨 Alertas Críticas", expanded=True):
                        for alerta in alertas['critical']:
                            st.error(
                                f"**{alerta['tipo']}**: {alerta['producto']}\n\n"
                                f"Categoría: {alerta['categoria']}\n\n"
                                f"{alerta['mensaje']}"
                            )
                
                if alertas['warning']:
                    with st.expander("⚠️ Advertencias", expanded=True):
                        for alerta in alertas['warning']:
                            st.warning(
                                f"**{alerta['tipo']}**: {alerta['producto']}\n\n"
                                f"Categoría: {alerta['categoria']}\n\n"
                                f"{alerta['mensaje']}"
                            )
            
            with col2:
                if alertas['opportunity']:
                    with st.expander("💡 Oportunidades", expanded=True):
                        for alerta in alertas['opportunity']:
                            st.info(
                                f"**{alerta['tipo']}**: {alerta['producto']}\n\n"
                                f"Categoría: {alerta['categoria']}\n\n"
                                f"{alerta['mensaje']}"
                            )

        # Tablas de información
        if not low_stock.empty or not recent_sales.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Productos con Stock Bajo")
                if not low_stock.empty:
                    st.dataframe(
                        low_stock.style.background_gradient(
                            subset=['stock_percentage'],
                            cmap='RdYlGn',
                            vmin=0,
                            vmax=100
                        ),
                        hide_index=True
                    )
                else:
                    st.info("No hay productos con stock bajo")
            
            with col2:
                st.subheader("Ventas Recientes")
                if not recent_sales.empty:
                    st.dataframe(
                        recent_sales.style.background_gradient(
                            subset=['total_venta'],
                            cmap='Blues'
                        ),
                        hide_index=True
                    )
                else:
                    st.info("No hay ventas registradas")

    except Exception as e:
        logger.error(f"Error en Dashboard: {e}")
        st.error(f"Error en el Dashboard: {str(e)}")
        st.info("Por favor, verifica la conexión a la base de datos y la disponibilidad de los datos")

elif page == "Productos":
    st.title("Gestión de Productos")
    
    try:
        # Lista de productos con búsqueda
        st.subheader("Catálogo de Productos")
        
        # Filtros
        col1, col2, col3 = st.columns(3)
        with col1:
            search_term = st.text_input("Buscar por nombre o SKU", "")
        with col2:
            categorias = pd.read_sql("SELECT DISTINCT categoria FROM productos", db.engine)
            categoria_filter = st.selectbox(
                "Filtrar por categoría", 
                ["Todas"] + categorias['categoria'].tolist()
            )
        with col3:
            stock_filter = st.selectbox(
                "Filtrar por stock",
                ["Todos", "Stock Bajo", "Sin Stock", "Stock Normal"]
            )
        
        # Construir consulta
        try:
            # Construir la consulta base sin concatenación
            conditions = []
            query_text = """
                SELECT 
                    p.*,
                    COALESCE(v.total_ventas, 0) as ventas_totales,
                    COALESCE(v.ultimo_mes, 0) as ventas_ultimo_mes
                FROM productos p
                LEFT JOIN (
                    SELECT 
                        producto_id,
                        COUNT(*) as total_ventas,
                        SUM(CASE WHEN fecha_venta >= CURRENT_DATE - INTERVAL '30 days'
                            THEN 1 ELSE 0 END) as ultimo_mes
                    FROM ventas
                    GROUP BY producto_id
                ) v ON p.id = v.producto_id
                WHERE 1=1
            """
            
            # Construir las condiciones y parámetros
            params = {}
            
            if search_term:
                conditions.append("(LOWER(p.nombre) LIKE LOWER(:search) OR LOWER(p.sku) LIKE LOWER(:search))")
                params['search'] = f"%{search_term}%"
            
            if categoria_filter != "Todas":
                conditions.append("p.categoria = :categoria")
                params['categoria'] = categoria_filter
            
            if stock_filter == "Stock Bajo":
                conditions.append("p.stock_actual <= p.stock_minimo")
            elif stock_filter == "Sin Stock":
                conditions.append("p.stock_actual = 0")
            elif stock_filter == "Stock Normal":
                conditions.append("p.stock_actual > p.stock_minimo")
            
            # Añadir condiciones a la consulta
            if conditions:
                query_text += " AND " + " AND ".join(conditions)
            
            # Crear objeto text() con la consulta completa y ejecutar
            query = text(query_text)
            productos = pd.read_sql(query, db.engine, params=params)
            
            if not productos.empty:
                st.dataframe(
                    productos.style.apply(lambda x: [
                        'background-color: #ffcccc' if x['stock_actual'] <= x['stock_minimo']
                        else 'background-color: #ccffcc' if x['stock_actual'] > x['stock_minimo'] * 2
                        else '' for i in x
                    ], axis=1),
                    hide_index=True
                )
                
                # Edición de productos
                st.subheader("Editar Producto")
                producto_id = st.selectbox(
                    "Selecciona un producto para editar",
                    options=productos['id'].tolist(),
                    format_func=lambda x: f"{productos[productos['id']==x]['nombre'].iloc[0]} (SKU: {productos[productos['id']==x]['sku'].iloc[0]})"
                )
                
                if producto_id:
                    producto_actual = productos[productos['id'] == producto_id].iloc[0]
                    
                    with st.form(key="form_editar_producto"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            nuevo_sku = st.text_input("SKU", value=producto_actual['sku'])
                            nuevo_nombre = st.text_input("Nombre", value=producto_actual['nombre'])
                            nueva_categoria = st.text_input("Categoría", value=producto_actual['categoria'])
                            
                        with col2:
                            nuevo_precio_compra = st.number_input(
                                "Precio de Compra", 
                                value=float(producto_actual['precio_compra']),
                                min_value=0.0,
                                format="%.2f"
                            )
                            nuevo_precio_venta = st.number_input(
                                "Precio de Venta",
                                value=float(producto_actual['precio_venta']),
                                min_value=0.0,
                                format="%.2f"
                            )
                            nuevo_stock_actual = st.number_input(
                                "Stock Actual",
                                value=int(producto_actual['stock_actual']),
                                min_value=0
                            )
                            nuevo_stock_minimo = st.number_input(
                                "Stock Mínimo",
                                value=int(producto_actual['stock_minimo']),
                                min_value=0
                            )
                        
                        if st.form_submit_button("Actualizar Producto"):
                            try:
                                db.update_product(
                                    producto_id,
                                    nuevo_sku,
                                    nuevo_nombre,
                                    nuevo_precio_compra,
                                    nuevo_precio_venta,
                                    nuevo_stock_actual,
                                    nuevo_stock_minimo,
                                    nueva_categoria
                                )
                                st.success("✅ Producto actualizado correctamente")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error al actualizar producto: {str(e)}")
            else:
                st.info("No se encontraron productos con los filtros seleccionados")
            
            # Formulario para añadir nuevo producto
            st.subheader("Añadir Nuevo Producto")
            with st.form(key="form_nuevo_producto"):
                col1, col2 = st.columns(2)
                
                with col1:
                    sku = st.text_input("SKU", key="nuevo_sku")
                    nombre = st.text_input("Nombre", key="nuevo_nombre")
                    categoria = st.text_input("Categoría", key="nueva_categoria")
                
                with col2:
                    precio_compra = st.number_input(
                        "Precio de Compra",
                        min_value=0.0,
                        format="%.2f",
                        key="nuevo_precio_compra"
                    )
                    precio_venta = st.number_input(
                        "Precio de Venta",
                        min_value=0.0,
                        format="%.2f",
                        key="nuevo_precio_venta"
                    )
                    stock_actual = st.number_input(
                        "Stock Actual",
                        min_value=0,
                        key="nuevo_stock_actual"
                    )
                    stock_minimo = st.number_input(
                        "Stock Mínimo",
                        min_value=0,
                        key="nuevo_stock_minimo"
                    )
                
                if st.form_submit_button("Añadir Producto"):
                    try:
                        db.add_product(
                            sku,
                            nombre,
                            precio_compra,
                            precio_venta,
                            stock_actual,
                            stock_minimo,
                            categoria
                        )
                        st.success("✅ Producto añadido correctamente")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al añadir producto: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error en consulta de productos: {e}")
            st.error(f"Error al cargar los productos: {str(e)}")

    except Exception as e:
        logger.error(f"Error en la página de productos: {e}")
        st.error(f"Error en la página de productos: {str(e)}")
        
elif page == "Ventas":
    st.title("Registro y Análisis de Ventas")
    
    try:
        tab1, tab2 = st.tabs(["Registro de Ventas", "Análisis de Ventas"])
        
        with tab1:
            # Formulario para registrar venta
            st.subheader("Registrar Nueva Venta")
            productos = pd.read_sql("""
                SELECT id, nombre, precio_venta, stock_actual, categoria 
                FROM productos 
                WHERE stock_actual > 0
                ORDER BY categoria, nombre
            """, db.engine)
            
            if not productos.empty:
                with st.form(key="venta_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        producto_id = st.selectbox(
                            "Producto",
                            options=productos['id'].tolist(),
                            format_func=lambda x: f"{productos[productos['id']==x]['nombre'].iloc[0]} "
                                                f"({productos[productos['id']==x]['categoria'].iloc[0]})"
                        )
                        
                        # Mostrar stock disponible
                        stock_actual = productos[productos['id'] == producto_id]['stock_actual'].iloc[0]
                        st.info(f"Stock disponible: {stock_actual} unidades")
                        
                    with col2:
                        cantidad = st.number_input(
                            "Cantidad",
                            min_value=1,
                            max_value=stock_actual,
                            value=1
                        )
                        precio_sugerido = productos[productos['id'] == producto_id]['precio_venta'].iloc[0]
                        precio_venta = st.number_input(
                            "Precio de Venta",
                            min_value=0.0,
                            value=float(precio_sugerido),
                            format="%.2f"
                        )
                    
                    # Preview de la venta
                    total_venta = cantidad * precio_venta
                    st.write(f"Total de venta: ${total_venta:.2f}")
                    
                    if st.form_submit_button("Registrar Venta"):
                        try:
                            db.register_sale(producto_id, cantidad, precio_venta)
                            st.success("✅ Venta registrada correctamente")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error al registrar venta: {str(e)}")
            else:
                st.warning("No hay productos disponibles para vender")
            
            # Historial de ventas recientes
            st.subheader("Últimas Ventas Registradas")
            ventas_recientes = pd.read_sql("""
                SELECT 
                    v.id,
                    p.nombre as producto,
                    p.categoria,
                    v.cantidad,
                    v.precio_venta,
                    (v.cantidad * v.precio_venta) as total,
                    v.fecha_venta
                FROM ventas v
                JOIN productos p ON v.producto_id = p.id
                ORDER BY v.fecha_venta DESC
                LIMIT 20
            """, db.engine)
            
            if not ventas_recientes.empty:
                st.dataframe(ventas_recientes, hide_index=True)
            else:
                st.info("No hay ventas registradas")
        
        with tab2:
            # Análisis de ventas con visualizaciones
            st.subheader("Análisis de Ventas")
            
            # Filtros de fecha
            col1, col2 = st.columns(2)
            with col1:
                fecha_inicio = st.date_input(
                    "Fecha inicial",
                    value=datetime.now() - timedelta(days=30)
                )
            with col2:
                fecha_fin = st.date_input(
                    "Fecha final",
                    value=datetime.now()
                )
            
            if st.button("Analizar Ventas"):
                # Cargar datos de ventas
                ventas_analisis = db.get_sales_report(fecha_inicio, fecha_fin)
                
                if not ventas_analisis.empty:
                    # Crear visualizaciones
                    sales_figures = visualizer.create_sales_analysis(ventas_analisis)
                    
                    # Mostrar visualizaciones
                    st.plotly_chart(sales_figures['trend'], use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(sales_figures['distribution'], use_container_width=True)
                    with col2:
                        st.plotly_chart(sales_figures['seasonality'], use_container_width=True)
                    
                    # Métricas de resumen
                    st.subheader("Resumen de Ventas")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Total Ventas",
                            f"${ventas_analisis['precio_venta'].sum():,.2f}"
                        )
                    with col2:
                        st.metric(
                            "Unidades Vendidas",
                            f"{ventas_analisis['cantidad'].sum():,.0f}"
                        )
                    with col3:
                        st.metric(
                            "Ticket Promedio",
                            f"${ventas_analisis['precio_venta'].mean():,.2f}"
                        )
                    with col4:
                        st.metric(
                            "Productos Diferentes",
                            f"{ventas_analisis['producto'].nunique():,.0f}"
                        )
                else:
                    st.info("No hay datos de ventas para el período seleccionado")
    except Exception as e:
        st.error(f"Error en la página de ventas: {str(e)}")

elif page == "Predicciones":
    st.title("Predicciones de Ventas")
    
    try:
        productos = pd.read_sql(
            text("""
                SELECT 
                    p.id, 
                    p.nombre,
                    p.categoria,
                    COUNT(v.id) as total_ventas
                FROM productos p
                LEFT JOIN ventas v ON p.id = v.producto_id
                GROUP BY p.id, p.nombre, p.categoria
                HAVING COUNT(v.id) > 0
                ORDER BY COUNT(v.id) DESC
            """),
            db.engine
        )
        
        if not productos.empty:
            # Selector de producto
            producto_id = st.selectbox(
                "Selecciona un producto",
                options=productos['id'].tolist(),
                format_func=lambda x: f"{productos[productos['id']==x]['nombre'].iloc[0]} "
                                    f"({productos[productos['id']==x]['categoria'].iloc[0]} - "
                                    f"{productos[productos['id']==x]['total_ventas'].iloc[0]} ventas)"
            )
            
            if st.button("Generar Predicción"):
                with st.spinner('Generando predicción...'):
                    try:
                        # Obtener datos históricos
                        ventas_historicas = db.get_product_sales(producto_id)
                        
                        if ventas_historicas is not None and not ventas_historicas.empty:
                            # Generar predicción
                            prediccion = predictor.predict_sales(ventas_historicas)
                            if prediccion is not None:
                                metricas = predictor.calculate_metrics(prediccion, ventas_historicas)
                                
                                # Obtener stock actual
                                stock_actual = 0
                                try:
                                    query = text("SELECT stock_actual FROM productos WHERE id = :id")
                                    with db.engine.connect() as conn:
                                        result = conn.execute(query, {"id": producto_id})
                                        stock_actual = result.scalar()
                                except Exception as e:
                                    logger.error(f"Error al obtener stock actual: {e}")
                                    st.error("Error al obtener el stock actual del producto")

                                # Visualizaciones
                                st.subheader("Predicción de Ventas - Próximos 3 Meses")
                                
                                # Crear figura de predicción
                                fig_pred = go.Figure()
                                
                                # Datos históricos
                                fig_pred.add_trace(
                                    go.Scatter(
                                        x=ventas_historicas['ds'],
                                        y=ventas_historicas['y'],
                                        name="Ventas Históricas",
                                        mode='lines+markers',
                                        line=dict(color='blue', width=1),
                                        marker=dict(size=6)
                                    )
                                )
                                
                                # Predicción
                                fig_pred.add_trace(
                                    go.Scatter(
                                        x=prediccion['ds'],
                                        y=prediccion['yhat'],
                                        name="Predicción",
                                        mode='lines',
                                        line=dict(color='green', width=2, dash='dash')
                                    )
                                )
                                
                                # Intervalo de confianza
                                fig_pred.add_trace(
                                    go.Scatter(
                                        x=prediccion['ds'].tolist() + prediccion['ds'].tolist()[::-1],
                                        y=prediccion['yhat_upper'].tolist() + prediccion['yhat_lower'].tolist()[::-1],
                                        fill='toself',
                                        fillcolor='rgba(0,176,246,0.2)',
                                        line=dict(color='rgba(255,255,255,0)'),
                                        name='Intervalo de Confianza',
                                        showlegend=True
                                    )
                                )
                                
                                fig_pred.update_layout(
                                    title="Predicción de Ventas",
                                    xaxis_title="Fecha",
                                    yaxis_title="Unidades Vendidas",
                                    hovermode='x unified'
                                )
                                
                                st.plotly_chart(fig_pred, use_container_width=True)

                                # Mostrar métricas y recomendaciones
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric(
                                        "Tendencia",
                                        f"{metricas['trend_change']:.1f}%",
                                        delta=metricas['trend_change']
                                    )
                                with col2:
                                    st.metric(
                                        "Stock Actual",
                                        f"{stock_actual} unidades"
                                    )
                                with col3:
                                    st.metric(
                                        "Stock Óptimo",
                                        f"{metricas['stock_recommendations']['optimal_stock']:.0f} unidades"
                                    )
                                with col4:
                                    st.metric(
                                        "Predicción Mensual",
                                        f"{metricas['monthly_predictions']['yhat'].mean():.1f} unidades/día"
                                    )

                                # Recomendaciones
                                st.subheader("Recomendaciones")
                                
                                # Stock
                                if stock_actual < metricas['stock_recommendations']['min_stock']:
                                    st.warning(
                                        f"⚠️ Stock bajo. Se recomienda comprar al menos "
                                        f"{metricas['stock_recommendations']['min_stock'] - stock_actual:.0f} unidades"
                                    )
                                elif stock_actual > metricas['stock_recommendations']['max_stock']:
                                    st.warning(
                                        f"⚠️ Stock alto. Se recomienda no comprar más hasta bajar de "
                                        f"{metricas['stock_recommendations']['optimal_stock']:.0f} unidades"
                                    )
                                else:
                                    st.success("✅ Nivel de stock óptimo")

                                # Patrones semanales
                                st.subheader("Patrones de Venta Semanal")
                                fig_pattern = px.bar(
                                    metricas['weekly_pattern'].reset_index(),
                                    x='weekday',
                                    y='yhat',
                                    title="Patrón de Ventas por Día de la Semana",
                                    labels={'weekday': 'Día', 'yhat': 'Ventas Promedio'}
                                )
                                st.plotly_chart(fig_pattern, use_container_width=True)
                            else:
                                st.error("No se pudieron generar las predicciones")
                        else:
                            st.warning("No hay suficientes datos históricos para este producto")
                    except Exception as e:
                        logger.error(f"Error en el procesamiento de predicciones: {e}")
                        st.error("Error al procesar las predicciones")
        else:
            st.warning("No hay productos con historial de ventas")
    
    except Exception as e:
        logger.error(f"Error en página de predicciones: {e}")
        st.error(f"Error en predicciones: {str(e)}")

elif page == "Reportes":
    st.title("Reportes y Exportación")
    
    report_type = st.selectbox(
        "Selecciona el tipo de reporte",
        ["Ventas", "Inventario", "Rendimiento"]
    )
    
    try:
        if report_type == "Ventas":
            st.subheader("Reporte de Ventas")
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Fecha inicial")
            with col2:
                end_date = st.date_input("Fecha final")
            
            if st.button("Generar Reporte de Ventas"):
                report_df = db.get_sales_report(start_date, end_date)
                
                if not report_df.empty:
                    # Visualizaciones
                    fig = visualizer.create_sales_analysis(report_df)
                    st.plotly_chart(fig['trend'], use_container_width=True)
                    
                    st.write("Resumen de Ventas:")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Ventas", f"${report_df['precio_venta'].sum():,.2f}")
                    with col2:
                        st.metric("Total Beneficio", f"${report_df['beneficio'].sum():,.2f}")
                    with col3:
                        st.metric("Margen Promedio", f"{report_df['margen_porcentaje'].mean():.1f}%")
                    
                    st.dataframe(report_df, hide_index=True)
                    
                    # Exportación
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        report_df.to_excel(writer, index=False, sheet_name='Ventas')
                        workbook = writer.book
                        worksheet = writer.sheets['Ventas']
                        
                        # Formato para moneda
                        money_fmt = workbook.add_format({'num_format': '$#,##0.00'})
                        percent_fmt = workbook.add_format({'num_format': '0.00%'})
                        
                        # Aplicar formatos
                        worksheet.set_column('F:F', 12, money_fmt)  # Precio venta
                        worksheet.set_column('G:G', 12, money_fmt)  # Beneficio
                        worksheet.set_column('H:H', 12, percent_fmt)  #worksheet.set_column('H:H', 12, percent_fmt)  # Margen
                        
                    st.download_button(
                        label="📥 Descargar Reporte Excel",
                        data=output.getvalue(),
                        file_name=f"reporte_ventas_{start_date}_{end_date}.xlsx",
                        mime="application/vnd.ms-excel"
                    )
                else:
                    st.info("No hay datos de ventas para el período seleccionado")
        
        elif report_type == "Inventario":
            st.subheader("Reporte de Inventario")
            
            report_df = db.get_inventory_report()
            
            if not report_df.empty:
                # Visualizaciones de inventario
                inventory_figures = visualizer.create_inventory_analysis(report_df)
                
                # Mostrar gráficos
                st.plotly_chart(inventory_figures['stock_dist'], use_container_width=True)
                st.plotly_chart(inventory_figures['rotation'], use_container_width=True)
                
                # Métricas de resumen
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Valor Total Inventario", 
                             f"${report_df['valor_inventario'].sum():,.2f}")
                with col2:
                    st.metric("Total Productos", 
                             len(report_df))
                with col3:
                    st.metric("Ventas Totales", 
                             f"{report_df['ventas_totales'].sum():,.0f}")
                
                # Mostrar tabla
                st.dataframe(report_df, hide_index=True)
                
                # Exportar
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    report_df.to_excel(writer, index=False, sheet_name='Inventario')
                    workbook = writer.book
                    worksheet = writer.sheets['Inventario']
                    
                    # Formato Excel
                    money_fmt = workbook.add_format({'num_format': '$#,##0.00'})
                    worksheet.set_column('E:G', 12, money_fmt)
                
                st.download_button(
                    label="📥 Descargar Reporte Excel",
                    data=output.getvalue(),
                    file_name=f"reporte_inventario_{datetime.now().date()}.xlsx",
                    mime="application/vnd.ms-excel"
                )
            else:
                st.info("No hay datos de inventario disponibles")
        
        elif report_type == "Rendimiento":
            st.subheader("Reporte de Rendimiento")
            
            periodo = st.selectbox(
                "Selecciona el periodo",
                ["day", "week", "month", "quarter", "year"],
                format_func=lambda x: {
                    "day": "Diario",
                    "week": "Semanal",
                    "month": "Mensual",
                    "quarter": "Trimestral",
                    "year": "Anual"
                }[x]
            )
            
            report_df = db.get_performance_report(periodo)
            
            if not report_df.empty:
                # Visualización de tendencias
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=report_df['periodo'],
                    y=report_df['ingresos_totales'],
                    name='Ingresos',
                    line=dict(color='#2ecc71', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=report_df['periodo'],
                    y=report_df['beneficio_total'],
                    name='Beneficio',
                    line=dict(color='#3498db', width=2)
                ))
                
                fig.update_layout(
                    title="Tendencia de Rendimiento",
                    xaxis_title="Periodo",
                    yaxis_title="Valor ($)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Métricas de resumen
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Promedio de Ingresos", 
                        f"${report_df['ingresos_totales'].mean():,.2f}"
                    )
                with col2:
                    st.metric(
                        "Promedio de Beneficio", 
                        f"${report_df['beneficio_total'].mean():,.2f}"
                    )
                with col3:
                    st.metric(
                        "Productos Promedio", 
                        f"{report_df['productos_vendidos'].mean():,.0f}"
                    )
                
                # Mostrar tabla
                st.dataframe(report_df, hide_index=True)
                
                # Exportar
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    report_df.to_excel(writer, index=False, sheet_name='Rendimiento')
                    workbook = writer.book
                    worksheet = writer.sheets['Rendimiento']
                    
                    money_fmt = workbook.add_format({'num_format': '$#,##0.00'})
                    worksheet.set_column('D:E', 12, money_fmt)
                
                st.download_button(
                    label="📥 Descargar Reporte Excel",
                    data=output.getvalue(),
                    file_name=f"reporte_rendimiento_{periodo}_{datetime.now().date()}.xlsx",
                    mime="application/vnd.ms-excel"
                )
            else:
                st.info("No hay datos de rendimiento disponibles")
    
    except Exception as e:
        st.error(f"Error en la página de reportes: {str(e)}")

elif page == "Métricas":
    st.title("Métricas de Rendimiento")
    
    try:
        # Métricas generales
        metricas = pd.read_sql("""
            WITH metricas_producto AS (
                SELECT 
                    p.id,
                    p.nombre,
                    p.categoria,
                    p.stock_actual,
                    p.stock_minimo,
                    COUNT(v.id) as total_ventas,
                    COALESCE(SUM(v.cantidad), 0) as unidades_vendidas,
                    COALESCE(SUM(v.cantidad * v.precio_venta), 0) as ingresos_totales,
                    COALESCE(SUM(v.cantidad * (v.precio_venta - p.precio_compra)), 0) as beneficio_total
                FROM productos p
                LEFT JOIN ventas v ON p.id = v.producto_id
                GROUP BY p.id, p.nombre, p.categoria
            )
            SELECT 
                *,
                CASE 
                    WHEN stock_actual > 0 
                    THEN (unidades_vendidas::float / NULLIF(stock_actual, 0))
                    ELSE 0 
                END as rotacion,
                CASE 
                    WHEN ingresos_totales > 0 
                    THEN (beneficio_total / ingresos_totales * 100)
                    ELSE 0 
                END as margen_porcentaje
            FROM metricas_producto
            ORDER BY beneficio_total DESC
        """, db.engine)
        
        if not metricas.empty:
            # Visualizaciones de métricas
            
            # 1. Beneficios por categoría
            fig_beneficios = px.pie(
                metricas,
                values='beneficio_total',
                names='categoria',
                title='Distribución de Beneficios por Categoría'
            )
            st.plotly_chart(fig_beneficios, use_container_width=True)
            
            # 2. Matriz de rendimiento
            fig_matriz = px.scatter(
                metricas,
                x='rotacion',
                y='margen_porcentaje',
                size='ingresos_totales',
                color='categoria',
                hover_data=['nombre', 'unidades_vendidas'],
                title='Matriz de Rendimiento: Rotación vs Margen'
            )
            st.plotly_chart(fig_matriz, use_container_width=True)
            
            # Tabla de métricas
            st.dataframe(
                metricas.style.background_gradient(
                    subset=['beneficio_total', 'rotacion', 'margen_porcentaje'],
                    cmap='RdYlGn'
                ),
                hide_index=True
            )
            
            # Exportar métricas
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                metricas.to_excel(writer, index=False, sheet_name='Metricas')
                
                workbook = writer.book
                worksheet = writer.sheets['Metricas']
                
                money_fmt = workbook.add_format({'num_format': '$#,##0.00'})
                percent_fmt = workbook.add_format({'num_format': '0.00%'})
                
                worksheet.set_column('H:I', 12, money_fmt)
                worksheet.set_column('K:K', 12, percent_fmt)
            
            st.download_button(
                label="📥 Descargar Métricas Excel",
                data=output.getvalue(),
                file_name=f"metricas_{datetime.now().date()}.xlsx",
                mime="application/vnd.ms-excel"
            )
        else:
            st.info("No hay métricas disponibles")
    
    except Exception as e:
        st.error(f"Error en la página de métricas: {str(e)}")

elif page == "Clustering":
    pagina_clustering(db)

elif page == "Anomalías":
    pagina_anomalias(db)