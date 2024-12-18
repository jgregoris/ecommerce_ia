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
from src.inventory_assistant import InventoryAssistant
from src.pagina_asistente import pagina_asistente
import logging
import time

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
        assistant = InventoryAssistant(db)
        return db, predictor, visualizer, assistant
    except Exception as e:
        logger.error(f"Error al inicializar componentes: {e}")
        return None, None, None, None  # Asegurarse de retornar 4 valores

db, predictor, visualizer, assistant = init_components()

if db is None or predictor is None or visualizer is None or assistant is None:
    st.error("No se pudieron inicializar los componentes necesarios.")
    st.stop()

# Sidebar para navegación
with st.sidebar:
    st.title("Amazon Analytics")
    page = st.selectbox(
        "Navegación",
        ["Dashboard", "Productos", "Ventas", "Predicciones", "Reportes", "Métricas", "Clustering", "Anomalías", "Asistente"]
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
        # Inicializar visualizador
        visualizer = DashboardVisualizer()
        
        # Cargar datos del dashboard
        low_stock, recent_sales, metrics = db.load_dashboard_data()
        
        if not metrics.empty:
            # Métricas principales (mantener las existentes)
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
            # Dashboard principal con visualizaciones
            sales_data = db.get_sales_report()
            inventory_data = db.get_inventory_report()
            
            if not sales_data.empty and not inventory_data.empty:
                # Obtener predicciones si hay un producto seleccionado
                predictions_data = None
                if 'selected_product_id' in st.session_state:
                    historical_data = db.get_product_sales(st.session_state.selected_product_id)
                    if historical_data is not None:
                        predictions_data = predictor.predict_sales(historical_data)

                # Crear y mostrar el dashboard principal
                main_dashboard = visualizer.create_main_dashboard(
                    sales_data,
                    inventory_data,
                    predictions_data
                )
                st.plotly_chart(main_dashboard, use_container_width=True)

            # Análisis de ventas
            st.subheader("Análisis Detallado de Ventas")
            if not sales_data.empty:
                sales_figures = visualizer.create_sales_analysis(sales_data)
                
                # Mostrar visualizaciones de ventas
                st.plotly_chart(sales_figures['trend'], use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(sales_figures['distribution'], use_container_width=True)
                with col2:
                    st.plotly_chart(sales_figures['seasonality'], use_container_width=True)

            # Análisis de inventario
            st.subheader("Análisis de Inventario")
            if not inventory_data.empty:
                inventory_figures = visualizer.create_inventory_analysis(inventory_data)
                
                # Mostrar visualizaciones de inventario
                st.plotly_chart(inventory_figures['stock_dist'], use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(inventory_figures['rotation'], use_container_width=True)
                with col2:
                    st.plotly_chart(inventory_figures['value'], use_container_width=True)

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
        # Inicializar visualizador
        visualizer = DashboardVisualizer()
        
        # Pestañas principales
        tab_catalogo, tab_analisis, tab_edicion = st.tabs([
            "📋 Catálogo", 
            "📊 Análisis",
            "✏️ Edición"
        ])
        
        # Consulta base para obtener productos
        query_text = """
            SELECT 
                p.*,
                COALESCE(v.total_ventas, 0) as ventas_totales,
                COALESCE(v.ultimo_mes, 0) as ventas_ultimo_mes,
                CAST((p.stock_actual::float / NULLIF(p.stock_minimo, 0) * 100) AS DECIMAL(10,2)) as stock_percentage
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
        
        with tab_catalogo:
            st.subheader("Catálogo de Productos")
            
            # Filtros
            col1, col2, col3 = st.columns(3)
            with col1:
                search_term = st.text_input("🔍 Buscar por nombre o SKU", "")
            with col2:
                categorias = pd.read_sql("SELECT DISTINCT categoria FROM productos", db.engine)
                categoria_filter = st.selectbox(
                    "📑 Filtrar por categoría", 
                    ["Todas"] + categorias['categoria'].tolist()
                )
            with col3:
                stock_filter = st.selectbox(
                    "📦 Filtrar por stock",
                    ["Todos", "Stock Bajo", "Sin Stock", "Stock Normal"]
                )
            
            # Construir condiciones de filtrado
            conditions = []
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
            
            # Ejecutar consulta
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
            else:
                st.info("No se encontraron productos con los filtros seleccionados")

        with tab_analisis:
            st.subheader("Análisis de Productos")
            
            if not productos.empty:
                # Calcular valor del inventario
                productos['valor_inventario'] = productos['stock_actual'] * productos['precio_compra']
                
                # 1. Distribución de Stock por Categoría
                fig_stock = go.Figure()
                stock_by_cat = productos.groupby('categoria').agg({
                    'stock_actual': 'sum',
                    'stock_minimo': 'sum'
                }).reset_index()
                
                fig_stock.add_trace(go.Bar(
                    name='Stock Actual',
                    x=stock_by_cat['categoria'],
                    y=stock_by_cat['stock_actual'],
                    marker_color='#2ecc71'
                ))
                
                fig_stock.add_trace(go.Bar(
                    name='Stock Mínimo',
                    x=stock_by_cat['categoria'],
                    y=stock_by_cat['stock_minimo'],
                    marker_color='#e74c3c'
                ))
                
                fig_stock.update_layout(
                    title='Distribución de Stock por Categoría',
                    barmode='group'
                )
                st.plotly_chart(fig_stock, use_container_width=True)
                
                # 2. Análisis de Valor de Inventario
                col1, col2 = st.columns(2)
                with col1:
                    # Gráfico de torta del valor del inventario
                    fig_valor = px.pie(
                        productos,
                        values='valor_inventario',
                        names='categoria',
                        title='Distribución del Valor del Inventario'
                    )
                    st.plotly_chart(fig_valor, use_container_width=True)
                
                with col2:
                    # Top productos por valor
                    top_productos = productos.nlargest(10, 'valor_inventario')
                    fig_top = px.bar(
                        top_productos,
                        x='nombre',
                        y='valor_inventario',
                        title='Top 10 Productos por Valor de Inventario'
                    )
                    fig_top.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_top, use_container_width=True)
                
                # 3. Análisis de Rotación
                productos['rotacion'] = productos['ventas_totales'] / productos['stock_actual'].replace(0, 1)
                fig_rotacion = px.scatter(
                    productos,
                    x='stock_actual',
                    y='rotacion',
                    color='categoria',
                    size='valor_inventario',
                    hover_data=['nombre'],
                    title='Análisis de Rotación vs Stock'
                )
                st.plotly_chart(fig_rotacion, use_container_width=True)
                
                # 4. Métricas Agregadas
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Total Productos",
                        f"{len(productos):,}"
                    )
                with col2:
                    st.metric(
                        "Valor Total Inventario",
                        f"${productos['valor_inventario'].sum():,.2f}"
                    )
                with col3:
                    st.metric(
                        "Productos Sin Stock",
                        f"{len(productos[productos['stock_actual'] == 0]):,}"
                    )
                with col4:
                    st.metric(
                        "Productos Stock Bajo",
                        f"{len(productos[productos['stock_actual'] <= productos['stock_minimo']]):,}"
                    )
            else:
                st.info("No hay datos disponibles para el análisis")

        with tab_edicion:
            st.subheader("Edición de Productos")
            
            if not productos.empty:
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
        logger.error(f"Error en la página de productos: {e}")
        st.error(f"Error en la página de productos: {str(e)}")

elif page == "Ventas":
    st.title("Registro y Análisis de Ventas")
    
    try:
        # Inicializar visualizador
        visualizer = DashboardVisualizer()
        
        # Crear pestañas
        tab_registro, tab_analisis, tab_tendencias = st.tabs([
            "🛍️ Registro de Ventas", 
            "📊 Análisis",
            "📈 Tendencias"
        ])
        
        # Dentro del tab_registro, reemplazar la sección del formulario de venta por:

        with tab_registro:
            st.subheader("Registrar Nueva Venta")
            
            # Obtener productos disponibles
            productos = pd.read_sql("""
                SELECT 
                    p.id, 
                    p.nombre, 
                    p.precio_venta, 
                    p.stock_actual, 
                    p.categoria,
                    COALESCE(v.ventas_ultimo_mes, 0) as ventas_ultimo_mes
                FROM productos p
                LEFT JOIN (
                    SELECT 
                        producto_id,
                        COUNT(*) as ventas_ultimo_mes
                    FROM ventas
                    WHERE fecha_venta >= CURRENT_DATE - INTERVAL '30 days'
                    GROUP BY producto_id
                ) v ON p.id = v.producto_id
                WHERE p.stock_actual > 0
                ORDER BY v.ventas_ultimo_mes DESC, p.categoria, p.nombre
            """, db.engine)
            
            if not productos.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    producto_id = st.selectbox(
                        "Producto",
                        options=productos['id'].tolist(),
                        format_func=lambda x: (
                            f"{productos[productos['id']==x]['nombre'].iloc[0]} - "
                            f"{productos[productos['id']==x]['categoria'].iloc[0]} "
                            f"({productos[productos['id']==x]['ventas_ultimo_mes'].iloc[0]} ventas último mes)"
                        ),
                        key='producto_venta'
                    )
                    
                    # Obtener información del producto seleccionado
                    producto_info = productos[productos['id'] == producto_id].iloc[0]
                    st.info(
                        f"📦 Stock disponible: {producto_info['stock_actual']} unidades\n\n"
                        f"💰 Precio de venta: ${producto_info['precio_venta']:.2f}"
                    )
                    
                with col2:
                    cantidad = st.number_input(
                        "Cantidad",
                        min_value=1,
                        max_value=int(producto_info['stock_actual']),
                        value=1
                    )
                    
                    # Mostrar el precio de venta como texto, no como input
                    st.text(f"Precio de Venta: ${producto_info['precio_venta']:.2f}")
                    
                # Preview de la venta con cálculos
                total_venta = cantidad * producto_info['precio_venta']
                
                # Botón de registro fuera del formulario para mejor control
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Venta", f"${total_venta:.2f}")
                with col2:
                    st.metric("Precio Unitario", f"${producto_info['precio_venta']:.2f}")
                with col3:
                    st.metric("Cantidad", f"{cantidad} unidades")
                
                if st.button("Registrar Venta", key="btn_registrar_venta"):
                    try:
                        db.register_sale(producto_id, cantidad, producto_info['precio_venta'])
                        st.success("✅ Venta registrada correctamente")
                        time.sleep(1)  # Pequeña pausa para mostrar el mensaje
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error al registrar venta: {str(e)}")
                
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
                        v.fecha_venta,
                        ((p.precio_venta - v.precio_venta) * v.cantidad) as descuento_aplicado
                    FROM ventas v
                    JOIN productos p ON v.producto_id = p.id
                    ORDER BY v.fecha_venta DESC
                    LIMIT 20
                """, db.engine)
                
                if not ventas_recientes.empty:
                    # Resumen de ventas recientes
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Total Ventas Recientes",
                            f"${ventas_recientes['total'].sum():,.2f}"
                        )
                    with col2:
                        st.metric(
                            "Unidades Vendidas",
                            f"{ventas_recientes['cantidad'].sum():,}"
                        )
                    with col3:
                        st.metric(
                            "Descuentos Aplicados",
                            f"${ventas_recientes['descuento_aplicado'].sum():,.2f}"
                        )
                    
                    # Tabla de ventas recientes con formato
                    st.dataframe(
                        ventas_recientes.style.background_gradient(
                            subset=['total'],
                            cmap='Blues'
                        ),
                        hide_index=True
                    )
                else:
                    st.info("No hay ventas registradas")
            else:
                st.warning("No hay productos disponibles para vender")
        
        with tab_analisis:
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
            
            if st.button("Analizar Ventas", key="analizar_ventas"):
                ventas_analisis = db.get_sales_report(fecha_inicio, fecha_fin)
                
                if not ventas_analisis.empty:
                    # Métricas de resumen
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "Total Ventas",
                            f"${ventas_analisis['total_venta'].sum():,.2f}"
                        )
                    with col2:
                        st.metric(
                            "Unidades Vendidas",
                            f"{ventas_analisis['cantidad'].sum():,.0f}"
                        )
                    with col3:
                        st.metric(
                            "Ticket Promedio",
                            f"${ventas_analisis['total_venta'].mean():,.2f}"
                        )
                    with col4:
                        st.metric(
                            "Margen Promedio",
                            f"{ventas_analisis['margen_porcentaje'].mean():,.1f}%"
                        )
                    
                    # Visualizaciones
                    sales_figures = visualizer.create_sales_analysis(ventas_analisis)
                    
                    # Gráfico de tendencia
                    st.plotly_chart(sales_figures['trend'], use_container_width=True)
                    
                    # Gráficos de distribución y estacionalidad
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(sales_figures['distribution'], use_container_width=True)
                    with col2:
                        st.plotly_chart(sales_figures['seasonality'], use_container_width=True)
                    
                    # Análisis por categoría
                    ventas_categoria = ventas_analisis.groupby('categoria').agg({
                        'total_venta': 'sum',
                        'cantidad': 'sum',
                        'beneficio': 'sum',
                        'margen_porcentaje': 'mean'
                    }).reset_index()
                    
                    st.subheader("Análisis por Categoría")
                    st.dataframe(
                        ventas_categoria.style.background_gradient(
                            subset=['total_venta', 'beneficio', 'margen_porcentaje'],
                            cmap='RdYlGn'
                        ),
                        hide_index=True
                    )
                else:
                    st.info("No hay datos de ventas para el período seleccionado")
        
        with tab_tendencias:
            st.subheader("Tendencias y Patrones")
            
            # Análisis de tendencias temporales
            ventas_tendencias = pd.read_sql("""
                WITH ventas_diarias AS (
                    SELECT 
                        DATE(fecha_venta) as fecha,
                        SUM(cantidad * precio_venta) as venta_total,
                        COUNT(DISTINCT producto_id) as productos_vendidos,
                        SUM(cantidad) as unidades_vendidas
                    FROM ventas
                    WHERE fecha_venta >= CURRENT_DATE - INTERVAL '90 days'
                    GROUP BY DATE(fecha_venta)
                )
                SELECT 
                    fecha,
                    venta_total,
                    productos_vendidos,
                    unidades_vendidas,
                    AVG(venta_total) OVER (ORDER BY fecha ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as media_movil_7d
                FROM ventas_diarias
                ORDER BY fecha
            """, db.engine)
            
            if not ventas_tendencias.empty:
                # Gráfico de tendencias
                fig_tendencias = go.Figure()
                
                # Ventas diarias
                fig_tendencias.add_trace(
                    go.Scatter(
                        x=ventas_tendencias['fecha'],
                        y=ventas_tendencias['venta_total'],
                        name='Ventas Diarias',
                        line=dict(color='#3498db', width=1)
                    )
                )
                
                # Media móvil
                fig_tendencias.add_trace(
                    go.Scatter(
                        x=ventas_tendencias['fecha'],
                        y=ventas_tendencias['media_movil_7d'],
                        name='Media Móvil (7 días)',
                        line=dict(color='#e74c3c', width=2, dash='dash')
                    )
                )
                
                fig_tendencias.update_layout(
                    title='Tendencia de Ventas (Últimos 90 días)',
                    xaxis_title='Fecha',
                    yaxis_title='Venta Total ($)',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_tendencias, use_container_width=True)
                
                # Métricas de tendencia
                ultimo_mes = ventas_tendencias.tail(30)['venta_total'].mean()
                mes_anterior = ventas_tendencias.iloc[-60:-30]['venta_total'].mean()
                variacion = ((ultimo_mes - mes_anterior) / mes_anterior * 100)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Promedio Últimos 30 días",
                        f"${ultimo_mes:,.2f}",
                        f"{variacion:+.1f}% vs mes anterior"
                    )
                with col2:
                    st.metric(
                        "Máximo Diario",
                        f"${ventas_tendencias['venta_total'].max():,.2f}"
                    )
                with col3:
                    st.metric(
                        "Mínimo Diario",
                        f"${ventas_tendencias['venta_total'].min():,.2f}"
                    )
                
                # Análisis de estacionalidad
                ventas_tendencias['dia_semana'] = pd.to_datetime(ventas_tendencias['fecha']).dt.day_name()
                patron_semanal = ventas_tendencias.groupby('dia_semana')['venta_total'].mean().reset_index()
                
                fig_patron = px.bar(
                    patron_semanal,
                    x='dia_semana',
                    y='venta_total',
                    title='Patrón de Ventas por Día de la Semana'
                )
                
                st.plotly_chart(fig_patron, use_container_width=True)
            else:
                st.info("No hay suficientes datos para análisis de tendencias")

    except Exception as e:
        logger.error(f"Error en la página de ventas: {e}")
        st.error(f"Error en la página de ventas: {str(e)}")

elif page == "Predicciones":
    st.title("Predicciones de Ventas")
    
    try:
        # Inicializar visualizador
        visualizer = DashboardVisualizer()
        
        # Crear pestañas
        tab_productos, tab_categorias = st.tabs([
            "🏷️ Predicción por Producto",
            "📊 Predicción por Categoría"
        ])
        
        with tab_productos:
            # Obtener productos con historial de ventas
            productos = pd.read_sql(
                text("""
                    SELECT 
                        p.id, 
                        p.nombre,
                        p.categoria,
                        p.stock_actual,
                        COUNT(v.id) as total_ventas,
                        AVG(v.cantidad) as promedio_ventas,
                        MAX(v.fecha_venta) as ultima_venta
                    FROM productos p
                    LEFT JOIN ventas v ON p.id = v.producto_id
                    GROUP BY p.id, p.nombre, p.categoria, p.stock_actual
                    HAVING COUNT(v.id) > 0
                    ORDER BY COUNT(v.id) DESC
                """),
                db.engine
            )
            
            if not productos.empty:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    producto_id = st.selectbox(
                        "Selecciona un producto",
                        options=productos['id'].tolist(),
                        format_func=lambda x: (
                            f"{productos[productos['id']==x]['nombre'].iloc[0]} - "
                            f"{productos[productos['id']==x]['categoria'].iloc[0]} "
                            f"({productos[productos['id']==x]['total_ventas'].iloc[0]} ventas)"
                        )
                    )
                    
                    # Mostrar información del producto seleccionado
                    producto_info = productos[productos['id'] == producto_id].iloc[0]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Stock Actual",
                            f"{producto_info['stock_actual']} unidades"
                        )
                    with col2:
                        st.metric(
                            "Promedio Ventas",
                            f"{producto_info['promedio_ventas']:.1f} unidades/venta"
                        )
                    with col3:
                        st.metric(
                            "Última Venta",
                            producto_info['ultima_venta'].strftime("%Y-%m-%d")
                        )
                
                with col2:
                    dias_prediccion = st.selectbox(
                        "Período de predicción",
                        options=[30, 60, 90],
                        format_func=lambda x: f"{x} días"
                    )
                
                if st.button("Generar Predicción", key="btn_prediccion"):
                    with st.spinner('Generando predicción...'):
                        # Obtener datos históricos
                        ventas_historicas = db.get_product_sales(producto_id)
                        
                        if ventas_historicas is not None and not ventas_historicas.empty:
                            # Generar predicción
                            prediccion = predictor.predict_sales(ventas_historicas, dias_prediccion)
                            if prediccion is not None:
                                metricas = predictor.calculate_metrics(prediccion, ventas_historicas)
                                
                                # 1. Gráfico de predicción
                                fig_pred = visualizer.create_forecast_visualization(
                                    ventas_historicas,
                                    prediccion
                                )
                                st.plotly_chart(fig_pred, use_container_width=True)
                                
                                # 2. Métricas y recomendaciones
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric(
                                        "Tendencia",
                                        f"{metricas['trend_change']:.1f}%" if abs(metricas['trend_change']) != float('inf') else "0%",
                                        delta=float(metricas['trend_change']) if abs(metricas['trend_change']) != float('inf') else 0
                                    )
                                with col2:
                                    st.metric(
                                        "Stock Óptimo",
                                        f"{metricas['stock_recommendations']['optimal_stock']:.0f} unidades"
                                        if metricas['stock_recommendations']['optimal_stock'] != float('inf') 
                                        else "N/A"
                                    )
                                with col3:
                                    venta_diaria = metricas['monthly_predictions']['yhat'].mean()
                                    st.metric(
                                        "Venta Diaria Esperada",
                                        f"{venta_diaria:.1f} unidades" if venta_diaria != float('inf') else "0 unidades"
                                    )
                                with col4:
                                    if producto_info['stock_actual'] > 0 and venta_diaria > 0:
                                        dias_stock = producto_info['stock_actual'] / venta_diaria
                                        st.metric(
                                            "Días de Stock",
                                            f"{dias_stock:.0f} días"
                                        )
                                    else:
                                        st.metric(
                                            "Días de Stock",
                                            "0 días"
                                        )
                                
                                # 3. Recomendaciones
                                with st.expander("📋 Recomendaciones", expanded=True):
                                    if producto_info['stock_actual'] < metricas['stock_recommendations']['min_stock']:
                                        st.warning(
                                            f"⚠️ Stock bajo. Se recomienda comprar al menos "
                                            f"{metricas['stock_recommendations']['min_stock'] - producto_info['stock_actual']:.0f} unidades"
                                        )
                                    elif producto_info['stock_actual'] > metricas['stock_recommendations']['max_stock']:
                                        st.warning(
                                            f"⚠️ Stock alto. Se recomienda no comprar más hasta bajar de "
                                            f"{metricas['stock_recommendations']['optimal_stock']:.0f} unidades"
                                        )
                                    else:
                                        st.success("✅ Nivel de stock óptimo")

                                # 4. Patrones de venta
                                st.subheader("Patrones de Venta")
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Patrón semanal
                                    if not metricas['weekly_pattern'].empty and 'dia' in metricas['weekly_pattern'].columns and 'ventas_promedio' in metricas['weekly_pattern'].columns:
                                        fig_pattern = px.bar(
                                            metricas['weekly_pattern'],
                                            x='dia',
                                            y='ventas_promedio',
                                            title="Ventas Promedio por Día de la Semana",
                                            labels={
                                                'dia': 'Día',
                                                'ventas_promedio': 'Ventas Promedio'
                                            }
                                        )
                                        fig_pattern.update_layout(
                                            showlegend=False,
                                            xaxis_tickangle=-45
                                        )
                                        st.plotly_chart(fig_pattern, use_container_width=True)
                                        
                                        # Mostrar tabla con detalles solo si tenemos todas las columnas necesarias
                                        if 'conteo' in metricas['weekly_pattern'].columns:
                                            st.markdown("### Detalles por Día")
                                            tabla_pattern = metricas['weekly_pattern'][['dia', 'ventas_promedio', 'conteo']]
                                            tabla_pattern = tabla_pattern.rename(columns={
                                                'dia': 'Día',
                                                'ventas_promedio': 'Promedio de Ventas',
                                                'conteo': 'Cantidad de Datos'
                                            })
                                            st.dataframe(
                                                tabla_pattern.style.format({
                                                    'Promedio de Ventas': '{:.2f}'
                                                }),
                                                hide_index=True
                                            )
                                    else:
                                        st.info("No hay suficientes datos para mostrar el patrón semanal")

                                with col2:
                                    # Distribución de ventas
                                    fig_dist = go.Figure()
                                    fig_dist.add_trace(go.Histogram(
                                        x=ventas_historicas['y'],
                                        name='Distribución de Ventas',
                                        nbinsx=20,
                                        showlegend=False
                                    ))
                                    fig_dist.update_layout(
                                        title="Distribución de Ventas Diarias",
                                        xaxis_title="Unidades Vendidas",
                                        yaxis_title="Frecuencia"
                                    )
                                    st.plotly_chart(fig_dist, use_container_width=True)

                                # 5. Guardar predicción
                                if st.button("Guardar Predicción"):
                                    try:
                                        prediccion_id = db.save_prediction(producto_id, prediccion)
                                        db.save_metrics(producto_id, metricas)
                                        st.success("✅ Predicción guardada correctamente")
                                    except Exception as e:
                                        st.error(f"Error al guardar predicción: {str(e)}")
                            else:
                                st.error("No se pudieron generar las predicciones")
                        else:
                            st.warning("No hay suficientes datos históricos para este producto")
            else:
                st.warning("No hay productos con historial de ventas")
        
        with tab_categorias:
            st.info("🚧 Funcionalidad en desarrollo. Próximamente disponible.")

    except Exception as e:
        logger.error(f"Error en página de predicciones: {e}")
        st.error(f"Error en predicciones: {str(e)}")

elif page == "Reportes":
    st.title("Reportes y Exportación")
    
    try:
        # Tabs para mejor organización
        tab_ventas, tab_inventario, tab_rendimiento = st.tabs([
            "📊 Reporte de Ventas",
            "📦 Reporte de Inventario",
            "📈 Reporte de Rendimiento"
        ])
        
        with tab_ventas:
            st.subheader("Análisis de Ventas")
            
            # Filtros de fecha y agrupación
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                start_date = st.date_input(
                    "Fecha inicial",
                    value=datetime.now() - timedelta(days=30)
                )
            with col2:
                end_date = st.date_input(
                    "Fecha final",
                    value=datetime.now()
                )
            with col3:
                agrupar_por = st.selectbox(
                    "Agrupar por",
                    ["Día", "Semana", "Mes"]
                )
            
            if st.button("Generar Reporte", key="gen_ventas"):
                with st.spinner("Generando reporte de ventas..."):
                    try:
                        report_df = db.get_sales_report(start_date, end_date)
                        
                        if not report_df.empty:
                            # Asegurar tipos de datos correctos
                            report_df['fecha_venta'] = pd.to_datetime(report_df['fecha_venta'])
                            
                            # Métricas principales
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric(
                                    "Total Ventas",
                                    f"${report_df['total_venta'].sum():,.2f}",
                                    help="Total de ingresos por ventas"
                                )
                            with col2:
                                st.metric(
                                    "Unidades Vendidas",
                                    f"{report_df['cantidad'].sum():,.0f}",
                                    help="Total de unidades vendidas"
                                )
                            with col3:
                                st.metric(
                                    "Ticket Promedio",
                                    f"${report_df['total_venta'].mean():,.2f}",
                                    help="Valor promedio por venta"
                                )
                            with col4:
                                margen = (report_df['beneficio'].sum() / report_df['total_venta'].sum() * 100)
                                st.metric(
                                    "Margen Promedio",
                                    f"{margen:.1f}%",
                                    help="Porcentaje de beneficio sobre ventas"
                                )
                            
                            # Preparar datos agrupados
                            if agrupar_por == "Día":
                                report_df['periodo'] = report_df['fecha_venta'].dt.date
                            elif agrupar_por == "Semana":
                                report_df['periodo'] = report_df['fecha_venta'].dt.to_period('W').astype(str)
                            else:  # Mes
                                report_df['periodo'] = report_df['fecha_venta'].dt.to_period('M').astype(str)
                            
                            ventas_agrupadas = report_df.groupby('periodo').agg({
                                'total_venta': 'sum',
                                'cantidad': 'sum',
                                'beneficio': 'sum'
                            }).reset_index()
                            
                            # Gráfico de tendencias
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=ventas_agrupadas['periodo'],
                                y=ventas_agrupadas['total_venta'],
                                name='Ventas',
                                line=dict(color='#2ecc71', width=2)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=ventas_agrupadas['periodo'],
                                y=ventas_agrupadas['beneficio'],
                                name='Beneficio',
                                line=dict(color='#3498db', width=2)
                            ))
                            
                            fig.update_layout(
                                title=f"Tendencia de Ventas y Beneficios por {agrupar_por}",
                                xaxis_title="Periodo",
                                yaxis_title="Valor ($)",
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Análisis por categoría
                            st.subheader("Análisis por Categoría")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Distribución de ventas
                                ventas_categoria = report_df.groupby('categoria')['total_venta'].sum()
                                fig_dist = px.pie(
                                    values=ventas_categoria.values,
                                    names=ventas_categoria.index,
                                    title="Distribución de Ventas por Categoría"
                                )
                                st.plotly_chart(fig_dist, use_container_width=True)
                            
                            with col2:
                                # Análisis de márgenes
                                margenes = report_df.groupby('categoria').agg({
                                    'beneficio': 'sum',
                                    'total_venta': 'sum'
                                })
                                margenes['margen'] = margenes['beneficio'] / margenes['total_venta'] * 100
                                
                                fig_margin = px.bar(
                                    x=margenes.index,
                                    y=margenes['margen'],
                                    title="Margen por Categoría (%)",
                                    labels={'x': 'Categoría', 'y': 'Margen (%)'}
                                )
                                st.plotly_chart(fig_margin, use_container_width=True)
                            
                            # Tabla resumen
                            st.subheader("Resumen Detallado")
                            
                            resumen = report_df.groupby('categoria').agg({
                                'total_venta': 'sum',
                                'cantidad': 'sum',
                                'beneficio': 'sum',
                                'producto': 'nunique'
                            }).reset_index()
                            
                            resumen = resumen.rename(columns={
                                'categoria': 'Categoría',
                                'total_venta': 'Ventas ($)',
                                'cantidad': 'Unidades',
                                'beneficio': 'Beneficio ($)',
                                'producto': 'Productos'
                            })
                            
                            st.dataframe(
                                resumen.style.format({
                                    'Ventas ($)': '${:,.2f}',
                                    'Beneficio ($)': '${:,.2f}'
                                })
                            )
                            
                            # Exportar
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                report_df.to_excel(writer, index=False, sheet_name='Detalle')
                                resumen.to_excel(writer, index=False, sheet_name='Resumen')
                                
                                # Formateo
                                workbook = writer.book
                                money_fmt = workbook.add_format({'num_format': '$#,##0.00'})
                                
                                for sheet in writer.sheets.values():
                                    sheet.set_column('E:G', 12, money_fmt)
                            
                            st.download_button(
                                label="📥 Descargar Reporte Excel",
                                data=output.getvalue(),
                                file_name=f"reporte_ventas_{start_date}_{end_date}.xlsx",
                                mime="application/vnd.ms-excel"
                            )
                        else:
                            st.info("No hay datos de ventas para el período seleccionado")
                            
                    except Exception as e:
                        st.error(f"Error al generar reporte de ventas: {str(e)}")
        
        with tab_inventario:
            st.subheader("Análisis de Inventario")
            
            try:
                inventory_df = db.get_inventory_report()
                
                if not inventory_df.empty:
                    # Métricas principales
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "Valor Total",
                            f"${inventory_df['valor_inventario'].sum():,.2f}"
                        )
                    with col2:
                        st.metric(
                            "Total Productos",
                            f"{len(inventory_df):,}"
                        )
                    with col3:
                        sin_stock = len(inventory_df[inventory_df['stock_actual'] == 0])
                        st.metric(
                            "Sin Stock",
                            f"{sin_stock:,}"
                        )
                    with col4:
                        stock_bajo = len(inventory_df[inventory_df['stock_actual'] <= 5])
                        st.metric(
                            "Stock Bajo",
                            f"{stock_bajo:,}"
                        )
                    
                    # Visualizaciones
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Valor por categoría
                        fig_valor = px.pie(
                            inventory_df,
                            values='valor_inventario',
                            names='categoria',
                            title="Distribución del Valor de Inventario"
                        )
                        st.plotly_chart(fig_valor, use_container_width=True)
                    
                    with col2:
                        # Stock por categoría
                        fig_stock = px.bar(
                            inventory_df.groupby('categoria')['stock_actual'].sum().reset_index(),
                            x='categoria',
                            y='stock_actual',
                            title="Stock por Categoría"
                        )
                        st.plotly_chart(fig_stock, use_container_width=True)
                    
                    # Tabla detallada
                    st.subheader("Detalle de Inventario")
                    
                    # Calcular días de stock
                    inventory_df['dias_stock'] = inventory_df['stock_actual'] / (
                        inventory_df['ventas_totales'].replace(0, 1) / 30
                    )
                    
                    tabla_inv = inventory_df[[
                        'nombre', 'categoria', 'stock_actual', 'valor_inventario',
                        'ventas_totales', 'dias_stock'
                    ]].copy()
                    
                    tabla_inv = tabla_inv.rename(columns={
                        'nombre': 'Producto',
                        'categoria': 'Categoría',
                        'stock_actual': 'Stock',
                        'valor_inventario': 'Valor ($)',
                        'ventas_totales': 'Ventas',
                        'dias_stock': 'Días de Stock'
                    })
                    
                    st.dataframe(
                        tabla_inv.style.format({
                            'Valor ($)': '${:,.2f}',
                            'Días de Stock': '{:.0f}'
                        })
                    )
                    
                    # Exportar
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        tabla_inv.to_excel(writer, index=False)
                        
                        workbook = writer.book
                        worksheet = writer.sheets['Sheet1']
                        
                        money_fmt = workbook.add_format({'num_format': '$#,##0.00'})
                        worksheet.set_column('D:D', 12, money_fmt)
                    
                    st.download_button(
                        label="📥 Descargar Reporte Excel",
                        data=output.getvalue(),
                        file_name=f"reporte_inventario_{datetime.now().date()}.xlsx",
                        mime="application/vnd.ms-excel"
                    )
                else:
                    st.info("No hay datos de inventario disponibles")
                    
            except Exception as e:
                st.error(f"Error al generar reporte de inventario: {str(e)}")
        
        with tab_rendimiento:
            st.subheader("Análisis de Rendimiento")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                periodo = st.selectbox(
                    "Periodo de análisis",
                    options=["day", "week", "month"],
                    format_func=lambda x: {
                        "day": "Diario",
                        "week": "Semanal",
                        "month": "Mensual"
                    }[x]
                )
            
            if st.button("Generar Reporte", key="gen_rendimiento"):
                with st.spinner("Generando reporte de rendimiento..."):
                    try:
                        perf_df = db.get_performance_report(periodo)
                        
                        if not perf_df.empty:
                            # Métricas principales
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Ingresos Promedio",
                                    f"${perf_df['ingresos_totales'].mean():,.2f}"
                                )
                            with col2:
                                st.metric(
                                    "Beneficio Promedio",
                                    f"${perf_df['beneficio_total'].mean():,.2f}"
                                )
                            with col3:
                                st.metric(
                                    "Margen Promedio",
                                    f"{perf_df['margen_porcentaje'].mean():.1f}%"
                                )
                            
                            # Gráfico de tendencias
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatter(
                                x=perf_df['periodo'],
                                y=perf_df['ingresos_totales'],
                                name='Ingresos',
                                line=dict(color='#2ecc71')
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=perf_df['periodo'],
                                y=perf_df['beneficio_total'],
                                name='Beneficio',
                                line=dict(color='#3498db')
                            ))
                            
                            fig.update_layout(
                                title="Tendencia de Rendimiento",
                                xaxis_title="Periodo",
                                yaxis_title="Valor ($)"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Tabla resumen
                            # Tabla resumen
                            tabla_perf = perf_df.copy()
                            tabla_perf['periodo'] = tabla_perf['periodo'].dt.strftime('%Y-%m-%d')
                            
                            tabla_perf = tabla_perf.rename(columns={
                                'periodo': 'Periodo',
                                'ingresos_totales': 'Ingresos ($)',
                                'beneficio_total': 'Beneficio ($)',
                                'margen_porcentaje': 'Margen (%)',
                                'productos_vendidos': 'Productos'
                            })
                            
                            st.dataframe(
                                tabla_perf.style.format({
                                    'Ingresos ($)': '${:,.2f}',
                                    'Beneficio ($)': '${:,.2f}',
                                    'Margen (%)': '{:.1f}%'
                                }).background_gradient(
                                    subset=['Margen (%)'],
                                    cmap='RdYlGn'
                                ),
                                hide_index=True
                            )
                            
                            # Análisis de márgenes
                            st.subheader("Análisis de Márgenes")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Evolución del margen
                                fig_margin = go.Figure()
                                
                                fig_margin.add_trace(go.Bar(
                                    x=perf_df['periodo'],
                                    y=perf_df['margen_porcentaje'],
                                    name='Margen',
                                    marker_color='#3498db'
                                ))
                                
                                fig_margin.update_layout(
                                    title="Evolución del Margen",
                                    xaxis_title="Periodo",
                                    yaxis_title="Margen (%)"
                                )
                                
                                st.plotly_chart(fig_margin, use_container_width=True)
                            
                            with col2:
                                # Distribución de márgenes
                                fig_hist = go.Figure()
                                
                                fig_hist.add_trace(go.Histogram(
                                    x=perf_df['margen_porcentaje'],
                                    nbinsx=20,
                                    marker_color='#3498db'
                                ))
                                
                                fig_hist.update_layout(
                                    title="Distribución de Márgenes",
                                    xaxis_title="Margen (%)",
                                    yaxis_title="Frecuencia"
                                )
                                
                                st.plotly_chart(fig_hist, use_container_width=True)
                            
                            # Exportación
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                tabla_perf.to_excel(writer, index=False, sheet_name='Rendimiento')
                                
                                workbook = writer.book
                                worksheet = writer.sheets['Rendimiento']
                                
                                # Formatos
                                money_fmt = workbook.add_format({'num_format': '$#,##0.00'})
                                percent_fmt = workbook.add_format({'num_format': '0.00%'})
                                
                                worksheet.set_column('B:C', 12, money_fmt)
                                worksheet.set_column('D:D', 12, percent_fmt)
                            
                            st.download_button(
                                label="📥 Descargar Reporte Excel",
                                data=output.getvalue(),
                                file_name=f"reporte_rendimiento_{periodo}_{datetime.now().date()}.xlsx",
                                mime="application/vnd.ms-excel"
                            )
                        else:
                            st.info("No hay datos de rendimiento disponibles")
                            
                    except Exception as e:
                        st.error(f"Error al generar reporte de rendimiento: {str(e)}")

    except Exception as e:
        logger.error(f"Error en la página de reportes: {e}")
        st.error(f"Error en reportes: {str(e)}")

elif page == "Métricas":
    st.title("Métricas de Rendimiento")
    
    try:
        # Métricas generales con KPIs avanzados
        metricas = pd.read_sql("""
            WITH ventas_periodo AS (
                SELECT 
                    p.id,
                    p.nombre,
                    p.categoria,
                    p.stock_actual,
                    p.stock_minimo,
                    p.precio_compra,
                    p.precio_venta,
                    CAST(p.stock_actual * p.precio_compra AS DECIMAL(10,2)) as valor_inventario,
                    COUNT(v.id) as total_ventas,
                    COALESCE(SUM(v.cantidad), 0) as unidades_vendidas,
                    COALESCE(SUM(v.cantidad * v.precio_venta), 0) as ingresos_totales,
                    COALESCE(SUM(v.cantidad * (v.precio_venta - p.precio_compra)), 0) as beneficio_total,
                    COUNT(DISTINCT DATE_TRUNC('month', v.fecha_venta)) as meses_con_ventas,
                    MAX(v.fecha_venta) as ultima_venta,
                    MIN(v.fecha_venta) as primera_venta,
                    COALESCE(SUM(v.cantidad) / 
                        NULLIF(EXTRACT(MONTH FROM AGE(MAX(v.fecha_venta), MIN(v.fecha_venta))) + 1, 0), 0
                    ) as ventas_mensuales_calc
                FROM productos p
                LEFT JOIN ventas v ON p.id = v.producto_id
                GROUP BY p.id, p.nombre, p.categoria, p.stock_actual, p.stock_minimo, p.precio_compra, p.precio_venta
            )
            SELECT 
                *,
                CAST((unidades_vendidas::float / NULLIF(stock_actual, 0)) AS DECIMAL(10,2)) as rotacion,
                CAST((beneficio_total / NULLIF(ingresos_totales, 0) * 100) AS DECIMAL(10,2)) as margen_porcentaje,
                CAST(ventas_mensuales_calc AS DECIMAL(10,2)) as ventas_mensuales,
                CAST((stock_actual::float / NULLIF(ventas_mensuales_calc, 0)) AS DECIMAL(10,2)) as meses_inventario,
                CAST(((precio_venta - precio_compra) / NULLIF(precio_compra, 0) * 100) AS DECIMAL(10,2)) as markup_porcentaje
            FROM ventas_periodo
            ORDER BY beneficio_total DESC;
            """
            , db.engine)
        
        if not metricas.empty:
            # KPIs Principales
            st.subheader("KPIs Principales")
            
            # Calcular métricas generales
            kpis = {
                "Margen Bruto Promedio": f"{metricas['margen_porcentaje'].mean():.1f}%",
                "Rotación Promedio": f"{metricas['rotacion'].mean():.2f}",
                "Markup Promedio": f"{metricas['markup_porcentaje'].mean():.1f}%",
                "Meses de Inventario": f"{metricas['meses_inventario'].median():.1f}",
                "Eficiencia Inventario": f"{(metricas['ingresos_totales'].sum() / metricas['valor_inventario'].sum()):.2f}x",
                "Productos sin Rotación": len(metricas[metricas['ventas_mensuales'] == 0]),
                "Top Categoría": metricas.groupby('categoria')['beneficio_total'].sum().idxmax()
            }
            
            # Mostrar KPIs en columnas
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Margen Bruto",
                    kpis["Margen Bruto Promedio"],
                    help="Margen bruto promedio de todos los productos"
                )
            with col2:
                st.metric(
                    "Rotación Inventario",
                    kpis["Rotación Promedio"],
                    help="Número de veces que rota el inventario"
                )
            with col3:
                st.metric(
                    "Markup Promedio",
                    kpis["Markup Promedio"],
                    help="Porcentaje de markup sobre el costo"
                )
            with col4:
                st.metric(
                    "Meses de Inventario",
                    kpis["Meses de Inventario"],
                    help="Meses estimados para agotar inventario"
                )

            # Análisis por Categoría
            st.subheader("Rendimiento por Categoría")
            
            # Calcular métricas por categoría
            metricas_categoria = metricas.groupby('categoria').agg({
                'ingresos_totales': 'sum',
                'beneficio_total': 'sum',
                'unidades_vendidas': 'sum',
                'margen_porcentaje': 'mean',
                'rotacion': 'mean',
                'valor_inventario': 'sum'
            }).round(2)
            
            metricas_categoria['roi'] = (metricas_categoria['beneficio_total'] / 
                                       metricas_categoria['valor_inventario'] * 100).round(2)
            
            # Mostrar tabla de métricas por categoría
            st.dataframe(
                metricas_categoria.style.format({
                    'ingresos_totales': '${:,.2f}',
                    'beneficio_total': '${:,.2f}',
                    'unidades_vendidas': '{:,.0f}',
                    'margen_porcentaje': '{:.1f}%',
                    'rotacion': '{:.2f}',
                    'valor_inventario': '${:,.2f}',
                    'roi': '{:.1f}%'
                }).background_gradient(
                    subset=['roi', 'rotacion', 'margen_porcentaje'],
                    cmap='RdYlGn'
                ),
                hide_index=False
            )

            # Top Productos por diferentes métricas
            st.subheader("Top Productos por Rendimiento")
            
            tab1, tab2, tab3 = st.tabs([
                "🔝 Mayor Beneficio",
                "📈 Mayor Rotación",
                "⚠️ Atención Requerida"
            ])
            
            with tab1:
                top_beneficio = metricas.nlargest(10, 'beneficio_total')[
                    ['nombre', 'categoria', 'beneficio_total', 'margen_porcentaje', 'rotacion']
                ]
                
                st.dataframe(
                    top_beneficio.style.format({
                        'beneficio_total': '${:,.2f}',
                        'margen_porcentaje': '{:.1f}%',
                        'rotacion': '{:.2f}'
                    }),
                    hide_index=True
                )
            
            with tab2:
                top_rotacion = metricas[metricas['unidades_vendidas'] > 0].nlargest(10, 'rotacion')[
                    ['nombre', 'categoria', 'rotacion', 'unidades_vendidas', 'stock_actual']
                ]
                
                st.dataframe(
                    top_rotacion.style.format({
                        'rotacion': '{:.2f}',
                        'unidades_vendidas': '{:,.0f}',
                        'stock_actual': '{:,.0f}'
                    }),
                    hide_index=True
                )
            
            with tab3:
                # Productos que requieren atención
                atencion = metricas[
                    (metricas['stock_actual'] > metricas['stock_minimo'] * 2) |  # Sobre stock
                    (metricas['ventas_mensuales'] == 0) |  # Sin ventas
                    (metricas['margen_porcentaje'] < 10)    # Margen bajo
                ].copy()
                
                atencion['alerta'] = ''
                atencion.loc[atencion['stock_actual'] > atencion['stock_minimo'] * 2, 'alerta'] += '📦 Sobre stock '
                atencion.loc[atencion['ventas_mensuales'] == 0, 'alerta'] += '⚠️ Sin ventas '
                atencion.loc[atencion['margen_porcentaje'] < 10, 'alerta'] += '💰 Margen bajo '
                
                st.dataframe(
                    atencion[['nombre', 'categoria', 'alerta', 'stock_actual', 'ventas_mensuales', 'margen_porcentaje']
                    ].style.format({
                        'stock_actual': '{:,.0f}',
                        'ventas_mensuales': '{:.1f}',
                        'margen_porcentaje': '{:.1f}%'
                    }),
                    hide_index=True
                )

            # Exportación de métricas
            st.subheader("Exportar Métricas")
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Hoja de métricas generales
                metricas.to_excel(writer, sheet_name='Métricas Detalladas', index=False)
                
                # Hoja de métricas por categoría
                metricas_categoria.to_excel(writer, sheet_name='Métricas por Categoría')
                
                # Formato
                workbook = writer.book
                money_fmt = workbook.add_format({'num_format': '$#,##0.00'})
                percent_fmt = workbook.add_format({'num_format': '0.00%'})
                
                # Aplicar formatos
                for sheet in writer.sheets.values():
                    sheet.set_column('H:I', 12, money_fmt)
                    sheet.set_column('J:K', 12, percent_fmt)
            
            st.download_button(
                label="📥 Descargar Métricas Completas",
                data=output.getvalue(),
                file_name=f"metricas_rendimiento_{datetime.now().date()}.xlsx",
                mime="application/vnd.ms-excel"
            )

        else:
            st.info("No hay datos de métricas disponibles")
            
    except Exception as e:
        logger.error(f"Error en la página de métricas: {e}")
        st.error(f"Error en métricas: {str(e)}")
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

elif page == "Asistente":
    pagina_asistente(db, assistant)