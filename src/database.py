import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import numpy as np

class DatabaseManager:
    def __init__(self):
        self.DATABASE_URL = "postgresql://localhost:5432/amazon_analytics"
        try:
            self.engine = create_engine(self.DATABASE_URL)
        except Exception as e:
            print(f"Error al conectar con la base de datos: {e}")
            raise

    def get_product_sales(self, producto_id):
        """Obtiene el historial de ventas de un producto con formato para Prophet"""
        query = text("""
            WITH dates AS (
                SELECT generate_series(
                    CURRENT_DATE - INTERVAL '90 days',
                    CURRENT_DATE,
                    '1 day'::interval
                )::date AS ds
            )
            SELECT 
                dates.ds,
                COALESCE(SUM(v.cantidad), 0) as y
            FROM dates
            LEFT JOIN ventas v ON dates.ds = v.fecha_venta::date 
                AND v.producto_id = :producto_id
            GROUP BY dates.ds
            ORDER BY dates.ds
        """)
        
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params={'producto_id': producto_id})
                if df.empty:
                    # Crear datos sintéticos si no hay datos reales
                    dates = pd.date_range(
                        start=datetime.now() - timedelta(days=90),
                        end=datetime.now(),
                        freq='D'
                    )
                    df = pd.DataFrame({
                        'ds': dates,
                        'y': np.random.normal(10, 2, size=len(dates))
                    })
                    df['y'] = df['y'].clip(lower=0)  # Asegurar valores no negativos
                return df
        except Exception as e:
            print(f"Error en get_product_sales: {str(e)}")
            return None

    def get_alerts(self):
        """Obtiene todas las alertas activas"""
        alerts = {
            'critical': [],
            'warning': [],
            'opportunity': []
        }
        
        try:
            stock_alerts = pd.read_sql("""
                SELECT 
                    id,
                    nombre,
                    stock_actual,
                    stock_minimo,
                    categoria,
                    CASE 
                        WHEN stock_actual = 0 THEN 'critical'
                        WHEN stock_actual <= stock_minimo THEN 'warning'
                        WHEN stock_actual <= stock_minimo * 1.2 THEN 'opportunity'
                    END as alert_type
                FROM productos
                WHERE stock_actual <= stock_minimo * 1.2
                ORDER BY 
                    CASE 
                        WHEN stock_actual = 0 THEN 1
                        WHEN stock_actual <= stock_minimo THEN 2
                        ELSE 3
                    END
            """, self.engine)
            
            for _, row in stock_alerts.iterrows():
                alert = {
                    'id': row['id'],
                    'tipo': 'Stock',
                    'producto': row['nombre'],
                    'mensaje': f"Stock actual: {row['stock_actual']} unidades (Mínimo: {row['stock_minimo']})",
                    'categoria': row['categoria']
                }
                alerts[row['alert_type']].append(alert)
            
            return alerts
        except Exception as e:
            print(f"Error al obtener alertas: {str(e)}")
            return alerts

    def get_sales_report(self, start_date=None, end_date=None):
        """Obtiene reporte detallado de ventas"""
        query = text("""
            SELECT 
                p.nombre as producto,
                p.categoria,
                v.cantidad,
                v.precio_venta,
                v.fecha_venta,
                CAST((v.precio_venta * v.cantidad) AS DECIMAL(10,2)) as total_venta,
                CAST((v.precio_venta - p.precio_compra) AS DECIMAL(10,2)) as beneficio,
                CAST(((v.precio_venta - p.precio_compra) / NULLIF(v.precio_venta, 0) * 100) AS DECIMAL(10,2)) as margen_porcentaje
            FROM ventas v
            JOIN productos p ON v.producto_id = p.id
            WHERE (:start_date IS NULL OR v.fecha_venta >= :start_date)
            AND (:end_date IS NULL OR v.fecha_venta <= :end_date)
            ORDER BY v.fecha_venta DESC
        """)
        
        try:
            with self.engine.connect() as conn:
                return pd.read_sql(query, conn, params={'start_date': start_date, 'end_date': end_date})
        except Exception as e:
            print(f"Error en get_sales_report: {str(e)}")
            return pd.DataFrame()

    def get_inventory_report(self):
        """Obtiene reporte de inventario actual"""
        query = text("""
            SELECT 
                p.nombre,
                p.categoria,
                p.stock_actual,
                COALESCE(SUM(v.cantidad), 0) as ventas_totales,
                CAST(p.stock_actual * p.precio_compra AS DECIMAL(10,2)) as valor_inventario
            FROM productos p
            LEFT JOIN ventas v ON p.id = v.producto_id
            GROUP BY p.id, p.nombre, p.categoria, p.stock_actual, p.precio_compra
            ORDER BY p.categoria, p.nombre
        """)
        
        try:
            with self.engine.connect() as conn:
                return pd.read_sql(query, conn)
        except Exception as e:
            print(f"Error en get_inventory_report: {str(e)}")
            return pd.DataFrame()

    def load_dashboard_data(self):
        """Carga los datos para el dashboard principal"""
        try:
            low_stock = pd.read_sql("""
                WITH stock_calc AS (
                    SELECT 
                        id, 
                        sku, 
                        nombre, 
                        stock_actual, 
                        stock_minimo,
                        CASE 
                            WHEN stock_minimo = 0 THEN 100.0
                            ELSE (stock_actual::decimal * 100.0 / NULLIF(stock_minimo, 0))
                        END as stock_calc
                    FROM productos
                    WHERE stock_actual <= stock_minimo * 1.2
                )
                SELECT 
                    id,
                    sku,
                    nombre,
                    stock_actual,
                    stock_minimo,
                    CAST(stock_calc AS DECIMAL(10,2)) as stock_percentage
                FROM stock_calc
                ORDER BY stock_calc ASC
            """, self.engine)
            
            recent_sales = pd.read_sql("""
                SELECT 
                    p.nombre, 
                    v.cantidad, 
                    v.precio_venta,
                    CAST(v.precio_venta * v.cantidad AS DECIMAL(10,2)) as total_venta,
                    v.fecha_venta,
                    p.categoria
                FROM ventas v
                JOIN productos p ON v.producto_id = p.id
                ORDER BY v.fecha_venta DESC
                LIMIT 10
            """, self.engine)
            
            metrics = pd.read_sql("""
                WITH ventas_totales AS (
                    SELECT 
                        CAST(SUM(v.cantidad * v.precio_venta) AS DECIMAL(10,2)) as total_ventas,
                        COUNT(DISTINCT v.producto_id) as productos_vendidos,
                        SUM(v.cantidad) as unidades_vendidas
                    FROM ventas v
                    WHERE v.fecha_venta >= CURRENT_DATE - INTERVAL '30 days'
                )
                SELECT 
                    COUNT(DISTINCT p.id) as total_productos,
                    SUM(p.stock_actual) as stock_total,
                    CAST(SUM(p.stock_actual * p.precio_compra) AS DECIMAL(10,2)) as valor_inventario,
                    COALESCE(vt.total_ventas, 0) as total_ventas,
                    COALESCE(vt.productos_vendidos, 0) as productos_vendidos,
                    COALESCE(vt.unidades_vendidas, 0) as unidades_vendidas
                FROM productos p
                LEFT JOIN ventas_totales vt ON true
                GROUP BY vt.total_ventas, vt.productos_vendidos, vt.unidades_vendidas
            """, self.engine)
            
            return low_stock, recent_sales, metrics
        except Exception as e:
            print(f"Error al cargar datos del dashboard: {str(e)}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()