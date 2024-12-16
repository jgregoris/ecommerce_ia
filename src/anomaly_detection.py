import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import streamlit as st
import logging

logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self, database_manager):
        self.db = database_manager

    def detect_sales_anomalies(self, producto_id, window_size=30):
        """
        Detecta anomalías en las ventas de un producto
        """
        try:
            # Obtener datos de ventas históricas
            sales_data = self.db.get_product_sales(producto_id)
            
            if sales_data is None or len(sales_data) < window_size:
                logger.warning(f"Datos insuficientes para producto {producto_id}")
                return None, None
                
            # Asegurar que no hay valores nulos o infinitos
            sales_data = sales_data.fillna(0)
            sales_data['y'] = sales_data['y'].replace([np.inf, -np.inf], 0)
            
            # Calcular la media móvil para suavizar los datos
            sales_data['y_smooth'] = sales_data['y'].rolling(window=7, min_periods=1).mean()
            
            # Descomponer la serie temporal con los datos suavizados
            decomposition = seasonal_decompose(sales_data['y_smooth'], period=7, extrapolate_trend='freq')
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid
            
            # Rellenar valores NaN en el residual con 0
            residual = residual.fillna(0)
            
            # Detectar anomalías en el residual
            anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            anomalies = anomaly_detector.fit_predict(residual.to_numpy().reshape(-1, 1))
            
            # Crear dataframe con anomalías
            sales_data['anomaly'] = anomalies
            sales_data['anomaly_score'] = -anomaly_detector.score_samples(residual.to_numpy().reshape(-1, 1))
            
            # Guardar anomalías detectadas en la base de datos
            anomalias_detectadas = sales_data[sales_data['anomaly'] == -1]
            for _, row in anomalias_detectadas.iterrows():
                anomaly_data = {
                    'tipo': 'venta',
                    'valor_detectado': float(row['y']),
                    'valor_esperado': float(row['y_smooth']),
                    'score': float(row['anomaly_score']),
                    'descripcion': f"Anomalía detectada en ventas: {row['y']:.2f} unidades (esperado: {row['y_smooth']:.2f})"
                }
                try:
                    self.db.save_anomaly(producto_id, anomaly_data)
                except Exception as e:
                    logger.error(f"Error al guardar anomalía: {e}")
            
            # Crear visualización
            fig = go.Figure()
            
            # Ventas históricas
            fig.add_trace(
                go.Scatter(
                    x=sales_data['ds'],
                    y=sales_data['y'],
                    name="Ventas",
                    mode='lines',
                    line=dict(color='blue', width=1)
                )
            )
            
            # Anomalías
            if len(anomalias_detectadas) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=anomalias_detectadas['ds'],
                        y=anomalias_detectadas['y'],
                        mode='markers',
                        name='Anomalías',
                        marker=dict(color='red', size=10, symbol='x'),
                    )
                )
            
            # Actualizar layout
            fig.update_layout(
                title="Detección de Anomalías en Ventas",
                xaxis_title="Fecha",
                yaxis_title="Unidades Vendidas",
                showlegend=True,
                hovermode='x unified'
            )
            
            return sales_data, fig
            
        except Exception as e:
            logger.error(f"Error en detect_sales_anomalies: {e}")
            return None, None

    def detect_inventory_anomalies(self, producto_id):
        """
        Detecta anomalías en el nivel de stock de un producto
        """
        try:
            query = """
                SELECT fecha_venta::date as fecha, 
                       stock_actual,
                       LEAD(stock_actual) OVER (ORDER BY fecha_venta) as next_stock
                FROM productos p
                JOIN ventas v ON p.id = v.producto_id
                WHERE p.id = :producto_id
                ORDER BY fecha_venta
            """
            
            inventory_data = pd.read_sql(query, self.db.engine, params={'producto_id': producto_id})
            
            if inventory_data.empty:
                logger.warning(f"Sin datos de inventario para producto {producto_id}")
                return None, None
            
            # Calcular cambios bruscos en el stock
            inventory_data['stock_change'] = inventory_data['stock_actual'] - inventory_data['next_stock']
            
            # Detectar anomalías en los cambios de stock
            anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            anomalies = anomaly_detector.fit_predict(inventory_data['stock_change'].to_numpy().reshape(-1, 1))
            
            inventory_data['anomaly'] = anomalies
            inventory_data['anomaly_score'] = -anomaly_detector.score_samples(
                inventory_data['stock_change'].to_numpy().reshape(-1, 1)
            )
            
            # Guardar anomalías detectadas
            anomalias_detectadas = inventory_data[inventory_data['anomaly'] == -1]
            for _, row in anomalias_detectadas.iterrows():
                anomaly_data = {
                    'tipo': 'inventario',
                    'valor_detectado': float(row['stock_actual']),
                    'valor_esperado': float(row['next_stock']),
                    'score': float(row['anomaly_score']),
                    'descripcion': f"Cambio anormal en stock: {row['stock_change']:.0f} unidades"
                }
                try:
                    self.db.save_anomaly(producto_id, anomaly_data)
                except Exception as e:
                    logger.error(f"Error al guardar anomalía de inventario: {e}")
            
            # Crear visualización
            fig = go.Figure()
            
            # Stock normal
            fig.add_trace(
                go.Scatter(
                    x=inventory_data['fecha'],
                    y=inventory_data['stock_actual'],
                    name="Stock",
                    mode='lines+markers',
                    line=dict(color='blue', width=1),
                    marker=dict(size=6)
                )
            )
            
            # Anomalías
            if len(anomalias_detectadas) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=anomalias_detectadas['fecha'],
                        y=anomalias_detectadas['stock_actual'],
                        mode='markers',
                        name='Anomalías',
                        marker=dict(color='red', size=10, symbol='x'),
                    )
                )
            
            fig.update_layout(
                title="Detección de Anomalías en Inventario",
                xaxis_title="Fecha",
                yaxis_title="Stock",
                showlegend=True,
                hovermode='x unified'
            )
            
            return inventory_data, fig
            
        except Exception as e:
            logger.error(f"Error en detect_inventory_anomalies: {e}")
            return None, None

def pagina_anomalias(db):
    st.title("Detección de Anomalías")
    
    try:
        # Verificar conexión a la base de datos
        if not hasattr(db, 'engine'):
            st.error("Error: No se ha inicializado correctamente la conexión a la base de datos")
            return
        
        # Consultar productos
        productos = pd.read_sql("""
            SELECT p.id, p.nombre, 
                COUNT(v.id) as num_ventas,
                COUNT(DISTINCT DATE(v.fecha_venta)) as dias_con_ventas
            FROM productos p
            LEFT JOIN ventas v ON p.id = v.producto_id
            GROUP BY p.id, p.nombre
            HAVING COUNT(v.id) > 0
            ORDER BY COUNT(v.id) DESC
        """, db.engine)
        
        if productos.empty:
            st.warning("No hay productos con ventas para analizar")
            return
        
        # Inicializar detector
        detector = AnomalyDetector(db)
        
        # Selector de producto
        producto_id = st.selectbox(
            "Selecciona un producto",
            options=productos['id'].tolist(),
            format_func=lambda x: f"{productos[productos['id']==x]['nombre'].iloc[0]} "
                                f"(Ventas: {productos[productos['id']==x]['num_ventas'].iloc[0]})",
            key="anomalias_producto_selector"
        )
        
        if st.button("Analizar Anomalías", key="btn_analizar_anomalias"):
            with st.spinner('Analizando datos...'):
                # Análisis de ventas
                st.subheader("Anomalías en Ventas")
                sales_data, sales_fig = detector.detect_sales_anomalies(producto_id)
                
                if sales_data is not None and not sales_data.empty:
                    st.plotly_chart(sales_fig, use_container_width=True)
                    anomalias_ventas = sales_data[sales_data['anomaly'] == -1]
                    if not anomalias_ventas.empty:
                        st.write("Anomalías detectadas en ventas:")
                        st.dataframe(
                            anomalias_ventas[['ds', 'y', 'anomaly_score']].sort_values('anomaly_score', ascending=False),
                            hide_index=True
                        )
                    else:
                        st.info("No se detectaron anomalías en las ventas")
                else:
                    st.warning("No hay suficientes datos de ventas para realizar el análisis")
                
                # Análisis de inventario
                st.subheader("Anomalías en Stock")
                inventory_data, inventory_fig = detector.detect_inventory_anomalies(producto_id)
                
                if inventory_data is not None and not inventory_data.empty:
                    st.plotly_chart(inventory_fig, use_container_width=True)
                    anomalias_inventario = inventory_data[inventory_data['anomaly'] == -1]
                    if not anomalias_inventario.empty:
                        st.write("Anomalías detectadas en inventario:")
                        st.dataframe(
                            anomalias_inventario[['fecha', 'stock_actual', 'stock_change', 'anomaly_score']]
                            .sort_values('anomaly_score', ascending=False),
                            hide_index=True
                        )
                    else:
                        st.info("No se detectaron anomalías en el inventario")
                else:
                    st.warning("No hay suficientes datos de inventario para realizar el análisis")

    except Exception as e:
        logger.error(f"Error en página de anomalías: {e}")
        st.error(f"Error al analizar anomalías: {str(e)}")
        st.info("Por favor, verifica la conexión a la base de datos y la disponibilidad de los datos")