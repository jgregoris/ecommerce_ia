import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Clase para procesar y limpiar datos antes de visualización o análisis"""
    
    def __init__(self, database_manager):
        self.db = database_manager

    def clean_sales_data(self, data):
        """Limpia y prepara datos de ventas"""
        try:
            df = data.copy()
            
            # Convertir fechas
            df['fecha_venta'] = pd.to_datetime(df['fecha_venta'])
            
            # Eliminar valores negativos
            df['cantidad'] = df['cantidad'].clip(lower=0)
            df['precio_venta'] = df['precio_venta'].clip(lower=0)
            
            # Calcular campos adicionales
            df['total_venta'] = df['cantidad'] * df['precio_venta']
            
            # Ordenar por fecha
            df = df.sort_values('fecha_venta')
            
            return df
            
        except Exception as e:
            logger.error(f"Error en clean_sales_data: {e}")
            return data

    def clean_inventory_data(self, data):
        """Limpia y prepara datos de inventario"""
        try:
            df = data.copy()
            
            # Eliminar valores negativos
            df['stock_actual'] = df['stock_actual'].clip(lower=0)
            df['precio_compra'] = df['precio_compra'].clip(lower=0)
            df['precio_venta'] = df['precio_venta'].clip(lower=0)
            
            # Calcular campos derivados
            df['valor_inventario'] = df['stock_actual'] * df['precio_compra']
            df['margen_potencial'] = ((df['precio_venta'] - df['precio_compra']) / df['precio_venta']) * 100
            
            return df
            
        except Exception as e:
            logger.error(f"Error en clean_inventory_data: {e}")
            return data

    def prepare_prediction_data(self, historical_data, days=90):
        """Prepara datos para predicciones"""
        try:
            df = historical_data.copy()
            
            # Asegurar formato de fecha
            df['ds'] = pd.to_datetime(df['ds'])
            
            # Rellenar fechas faltantes
            date_range = pd.date_range(
                start=df['ds'].min(),
                end=df['ds'].max(),
                freq='D'
            )
            df = df.set_index('ds').reindex(date_range).reset_index()
            df = df.rename(columns={'index': 'ds'})
            
            # Rellenar valores nulos
            df['y'] = df['y'].fillna(0)
            
            # Eliminar valores negativos
            df['y'] = df['y'].clip(lower=0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error en prepare_prediction_data: {e}")
            return historical_data

    def aggregate_sales_data(self, data, freq='D'):
        """Agrega datos de ventas por frecuencia especificada"""
        try:
            df = data.copy()
            
            # Agrupar por fecha y categoría
            aggregated = df.groupby([
                pd.Grouper(key='fecha_venta', freq=freq),
                'categoria'
            ]).agg({
                'cantidad': 'sum',
                'total_venta': 'sum',
                'beneficio': 'sum'
            }).reset_index()
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error en aggregate_sales_data: {e}")
            return data

    def calculate_product_metrics(self, producto_id):
        """Calcula métricas detalladas para un producto"""
        try:
            # Obtener datos del producto
            query = """
                WITH metricas AS (
                    SELECT 
                        p.id,
                        p.nombre,
                        p.stock_actual,
                        p.precio_compra,
                        p.precio_venta,
                        COUNT(v.id) as total_ventas,
                        COALESCE(AVG(v.cantidad), 0) as venta_promedio,
                        COALESCE(SUM(v.cantidad * v.precio_venta), 0) as ingresos_totales,
                        COALESCE(SUM(v.cantidad * (v.precio_venta - p.precio_compra)), 0) as beneficio_total
                    FROM productos p
                    LEFT JOIN ventas v ON p.id = v.producto_id
                    WHERE p.id = :producto_id
                    GROUP BY p.id, p.nombre, p.stock_actual, p.precio_compra, p.precio_venta
                )
                SELECT 
                    *,
                    CASE 
                        WHEN stock_actual > 0 THEN total_ventas::float / stock_actual 
                        ELSE 0 
                    END as rotacion,
                    CASE 
                        WHEN ingresos_totales > 0 THEN (beneficio_total / ingresos_totales) * 100
                        ELSE 0
                    END as margen_porcentaje
                FROM metricas
            """
            
            with self.db.engine.connect() as conn:
                metrics = pd.read_sql(query, conn, params={'producto_id': producto_id})
                
                if not metrics.empty:
                    metrics = metrics.iloc[0].to_dict()
                else:
                    metrics = {}
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error en calculate_product_metrics: {e}")
            return {}

    def get_trend_analysis(self, producto_id, days=30):
        """Analiza tendencias de ventas para un producto"""
        try:
            query = """
                WITH daily_sales AS (
                    SELECT 
                        DATE_TRUNC('day', fecha_venta) as fecha,
                        SUM(cantidad) as ventas,
                        AVG(precio_venta) as precio_promedio
                    FROM ventas
                    WHERE producto_id = :producto_id
                    AND fecha_venta >= CURRENT_DATE - :days * INTERVAL '1 day'
                    GROUP BY DATE_TRUNC('day', fecha_venta)
                )
                SELECT 
                    fecha,
                    ventas,
                    precio_promedio,
                    AVG(ventas) OVER (ORDER BY fecha ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as media_movil_7d
                FROM daily_sales
                ORDER BY fecha
            """
            
            with self.db.engine.connect() as conn:
                trend_data = pd.read_sql(
                    query, 
                    conn, 
                    params={'producto_id': producto_id, 'days': days}
                )
                
                return trend_data
                
        except Exception as e:
            logger.error(f"Error en get_trend_analysis: {e}")
            return pd.DataFrame()
