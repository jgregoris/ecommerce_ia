import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import streamlit as st

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
                return None, plt.figure()
                
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
            # Crear dataframe con anomalías
            sales_data['anomaly'] = anomalies
            sales_data['anomaly_score'] = -anomaly_detector.score_samples(residual.to_numpy().reshape(-1, 1))
            
            # Visualizar resultados
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
            
            # Ventas históricas y anomalías
            ax1.plot(sales_data['ds'], sales_data['y'], label='Ventas', alpha=0.7)
            ax1.scatter(sales_data[sales_data['anomaly'] == -1]['ds'],
                       sales_data[sales_data['anomaly'] == -1]['y'],
                       color='red', label='Anomalías')
            ax1.set_title('Ventas Históricas y Anomalías')
            ax1.legend()
            
            # Descomposición
            ax2.plot(sales_data['ds'], trend, label='Tendencia')
            ax2.plot(sales_data['ds'], seasonal, label='Estacionalidad')
            ax2.legend()
            ax2.set_title('Descomposición de la Serie Temporal')
            
            # Scores de anomalías
            ax3.scatter(sales_data['ds'], sales_data['anomaly_score'])
            ax3.axhline(y=0.5, color='r', linestyle='--', label='Umbral')
            ax3.set_title('Scores de Anomalías')
            ax3.legend()
            
            plt.tight_layout()
            return sales_data, fig
            
        except Exception as e:
            print(f"Error en detect_sales_anomalies: {str(e)}")
            return None, plt.figure()

    def detect_inventory_anomalies(self, producto_id):
        """
        Detecta anomalías en el nivel de stock de un producto
        """
        try:
            # Obtener datos de stock históricos
            inventory_data = self.db.get_product_inventory(producto_id)
            
            if inventory_data is None or len(inventory_data) < 7:  # Mínimo de datos requeridos
                return None, plt.figure()
            
            # Asegurar que no hay valores nulos o infinitos
            inventory_data = inventory_data.fillna(method='ffill').fillna(0)
            inventory_data['stock_actual'] = inventory_data['stock_actual'].replace([np.inf, -np.inf], 0)
            
            # Detectar anomalías en el stock actual
            anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            anomalies = anomaly_detector.fit_predict(inventory_data['stock_actual'].to_numpy().reshape(-1, 1))
            
            # Crear dataframe con anomalías
            inventory_data['anomaly'] = anomalies
            inventory_data['anomaly_score'] = -anomaly_detector.score_samples(
                inventory_data['stock_actual'].to_numpy().reshape(-1, 1)
            )
            
            # Visualizar resultados
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Stock y anomalías
            ax1.plot(inventory_data['fecha_actualizacion'], inventory_data['stock_actual'], 
                    label='Stock', alpha=0.7)
            ax1.scatter(inventory_data[inventory_data['anomaly'] == -1]['fecha_actualizacion'],
                       inventory_data[inventory_data['anomaly'] == -1]['stock_actual'],
                       color='red', label='Anomalías')
            ax1.set_title(f'Stock y Anomalías - Producto {producto_id}')
            ax1.legend()
            
            # Scores de anomalías
            ax2.scatter(inventory_data['fecha_actualizacion'], inventory_data['anomaly_score'])
            ax2.axhline(y=0.5, color='r', linestyle='--', label='Umbral')
            ax2.set_title('Scores de Anomalías')
            ax2.legend()
            
            plt.tight_layout()
            return inventory_data, fig
            
        except Exception as e:
            print(f"Error en detect_inventory_anomalies: {str(e)}")
            return None, plt.figure()

def pagina_anomalias(db):
    st.title("Detección de Anomalías")
    
    try:
        # Debug: Verificar conexión a la base de datos
        st.write("Verificando conexión a la base de datos...")
        if not hasattr(db, 'engine'):
            st.error("Error: No se ha inicializado correctamente la conexión a la base de datos")
            return
            
        # Debug: Intentar consulta de productos
        st.write("Consultando productos...")
        try:
            productos = pd.read_sql("""
                SELECT p.id, p.nombre, 
                    COUNT(v.id) as num_ventas,
                    COUNT(DISTINCT DATE(v.fecha_venta)) as dias_con_ventas
                FROM productos p
                LEFT JOIN ventas v ON p.id = v.producto_id
                GROUP BY p.id, p.nombre
            """, db.engine)
            st.write(f"Productos encontrados: {len(productos)}")
        except Exception as e:
            st.error(f"Error al consultar productos: {str(e)}")
            return
            
        # Verificar si hay productos
        if productos.empty:
            st.warning("No hay productos disponibles para analizar")
            return
            
        # Inicializar el detector de anomalías
        try:
            anomaly_detector = AnomalyDetector(db)
            st.write("Detector de anomalías inicializado correctamente")
        except Exception as e:
            st.error(f"Error al inicializar detector de anomalías: {str(e)}")
            return
        
        # Selector de producto con key única y información adicional
        producto_id = st.selectbox(
            "Selecciona un producto",
            options=productos['id'].tolist(),
            format_func=lambda x: f"{productos[productos['id']==x]['nombre'].iloc[0]} " 
                                f"(Ventas: {productos[productos['id']==x]['num_ventas'].iloc[0]})",
            key="anomalias_producto_selector"
        )
        
        # Verificar si el producto tiene suficientes datos
        producto_info = productos[productos['id'] == producto_id].iloc[0]
        if producto_info['num_ventas'] < 7:
            st.warning(f"⚠️ Este producto tiene pocas ventas ({producto_info['num_ventas']}). "
                      "Se necesitan al menos 7 ventas para un análisis significativo.")
        
        if st.button("Analizar Anomalías", key="btn_analizar_anomalias"):
            with st.spinner('Analizando datos...'):
                try:
                    # Detección de anomalías en ventas
                    st.subheader("Anomalías en Ventas")
                    sales_data, sales_fig = anomaly_detector.detect_sales_anomalies(producto_id)
                    
                    if sales_data is not None and not sales_data.empty:
                        st.pyplot(sales_fig)
                        
                        # Mostrar tabla de anomalías en ventas
                        anomalias_ventas = sales_data[sales_data['anomaly'] == -1]
                        if not anomalias_ventas.empty:
                            st.write("Anomalías detectadas en ventas:")
                            st.dataframe(anomalias_ventas[['ds', 'y', 'anomaly_score']], 
                                       key="df_anomalias_ventas")
                        else:
                            st.info("No se detectaron anomalías en las ventas")
                    else:
                        st.warning("No hay suficientes datos de ventas para realizar el análisis")

                    # Detección de anomalías en stock
                    st.subheader("Anomalías en Stock")
                    inventory_data, inventory_fig = anomaly_detector.detect_inventory_anomalies(producto_id)
                    
                    if inventory_data is not None and not inventory_data.empty:
                        st.pyplot(inventory_fig)
                        
                        # Mostrar tabla de anomalías en inventario
                        anomalias_inventario = inventory_data[inventory_data['anomaly'] == -1]
                        if not anomalias_inventario.empty:
                            st.write("Anomalías detectadas en inventario:")
                            st.dataframe(anomalias_inventario[['fecha_actualizacion', 'stock_actual', 'anomaly_score']], 
                                       key="df_anomalias_inventario")
                        else:
                            st.info("No se detectaron anomalías en el inventario")
                    else:
                        st.warning("No hay suficientes datos de inventario para realizar el análisis")

                except Exception as e:
                    st.error(f"Error durante el análisis: {str(e)}")
                    st.exception(e)  # Esto mostrará el stack trace completo
                    st.info("Por favor, contacta al equipo de soporte con el error mostrado arriba")

    except Exception as e:
        st.error(f"Error general en la página de anomalías: {str(e)}")
        st.info("Por favor, verifica la conexión a la base de datos y la disponibilidad de los datos")