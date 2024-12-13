import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import numpy as np

class DatabaseManager:
    def __init__(self):
        self.DATABASE_URL = "postgresql://localhost:5432/amazon_analytics"
        self.engine = create_engine(self.DATABASE_URL)

    def add_product(self, sku, nombre, precio_compra, precio_venta, stock_actual, stock_minimo, categoria):
        """Añade un nuevo producto a la base de datos"""
        query = text("""
            INSERT INTO productos (sku, nombre, precio_compra, precio_venta, stock_actual, stock_minimo, categoria)
            VALUES (:sku, :nombre, :precio_compra, :precio_venta, :stock_actual, :stock_minimo, :categoria)
            RETURNING id
        """)
        with self.engine.connect() as conn:
            result = conn.execute(query, {
                'sku': sku,
                'nombre': nombre,
                'precio_compra': precio_compra,
                'precio_venta': precio_venta,
                'stock_actual': stock_actual,
                'stock_minimo': stock_minimo,
                'categoria': categoria
            })
            conn.commit()
            return result.scalar()

    def update_product(self, product_id, sku, nombre, precio_compra, precio_venta, 
                      stock_actual, stock_minimo, categoria):
        """Actualiza un producto existente"""
        query = text("""
            UPDATE productos 
            SET sku = :sku,
                nombre = :nombre,
                precio_compra = :precio_compra,
                precio_venta = :precio_venta,
                stock_actual = :stock_actual,
                stock_minimo = :stock_minimo,
                categoria = :categoria
            WHERE id = :product_id
        """)
        
        with self.engine.connect() as conn:
            conn.execute(query, {
                'product_id': product_id,
                'sku': sku,
                'nombre': nombre,
                'precio_compra': precio_compra,
                'precio_venta': precio_venta,
                'stock_actual': stock_actual,
                'stock_minimo': stock_minimo,
                'categoria': categoria
            })
            conn.commit()

    def register_sale(self, producto_id, cantidad, precio_venta):
        """Registra una venta y actualiza el stock"""
        with self.engine.connect() as conn:
            # Registrar la venta
            query_venta = text("""
                INSERT INTO ventas (producto_id, cantidad, precio_venta)
                VALUES (:producto_id, :cantidad, :precio_venta)
            """)
            
            # Actualizar stock
            query_stock = text("""
                UPDATE productos 
                SET stock_actual = stock_actual - :cantidad
                WHERE id = :producto_id
            """)
            
            conn.execute(query_venta, {
                'producto_id': producto_id,
                'cantidad': cantidad,
                'precio_venta': precio_venta
            })
            
            conn.execute(query_stock, {
                'producto_id': producto_id,
                'cantidad': cantidad
            })
            
            conn.commit()

    def get_product_sales(self, producto_id):
        """
        Obtiene el historial de ventas de un producto con formato adecuado para Prophet
        """
        query = text("""
            WITH dates AS (
                SELECT generate_series(
                    (SELECT MIN(fecha_venta)::date FROM ventas WHERE producto_id = :producto_id),
                    CURRENT_DATE,
                    '1 day'::interval
                ) AS fecha
            )
            SELECT 
                dates.fecha as ds,
                COALESCE(SUM(v.cantidad), 0) as y
            FROM dates
            LEFT JOIN ventas v ON dates.fecha::date = v.fecha_venta::date 
                AND v.producto_id = :producto_id
            GROUP BY dates.fecha
            ORDER BY dates.fecha
        """)
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={'producto_id': producto_id})
            
            # Asegurar que tenemos al menos 14 días de datos
            if len(df) < 14:
                # Generar datos sintéticos si no hay suficientes
                dates = pd.date_range(
                    start=datetime.now() - timedelta(days=30),
                    end=datetime.now(),
                    freq='D'
                )
                df = pd.DataFrame({
                    'ds': dates,
                    'y': np.random.normal(10, 2, size=len(dates))  # Datos sintéticos
                })
            
            return df

    def get_alerts(self):
        """Obtiene todas las alertas activas"""
        alerts = {
            'critical': [],
            'warning': [],
            'opportunity': []
        }
        
        # Alertas de stock bajo
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
        
        # Productos sin ventas recientes
        no_sales_alerts = pd.read_sql("""
            SELECT 
                p.id,
                p.nombre,
                p.categoria,
                MAX(v.fecha_venta) as ultima_venta,
                CURRENT_DATE - MAX(v.fecha_venta) as dias_sin_ventas
            FROM productos p
            LEFT JOIN ventas v ON p.id = v.producto_id
            GROUP BY p.id, p.nombre, p.categoria
            HAVING MAX(v.fecha_venta) < CURRENT_DATE - INTERVAL '30 days'
            OR MAX(v.fecha_venta) IS NULL
        """, self.engine)
        
        # Procesar alertas de stock
        for _, row in stock_alerts.iterrows():
            alert = {
                'id': row['id'],
                'tipo': 'Stock',
                'producto': row['nombre'],
                'mensaje': f"Stock actual: {row['stock_actual']} unidades (Mínimo: {row['stock_minimo']})",
                'categoria': row['categoria']
            }
            alerts[row['alert_type']].append(alert)
        
        # Procesar alertas de ventas
        for _, row in no_sales_alerts.iterrows():
            alert = {
                'id': row['id'],
                'tipo': 'Ventas',
                'producto': row['nombre'],
                'mensaje': f"Sin ventas durante {row['dias_sin_ventas']} días",
                'categoria': row['categoria']
            }
            alerts['warning'].append(alert)
        
        return alerts

    def get_sales_report(self, start_date=None, end_date=None):
        """Obtiene reporte detallado de ventas"""
        query = text("""
            SELECT 
                p.nombre as producto,
                p.categoria,
                p.precio_compra,
                v.precio_venta,
                v.cantidad,
                v.fecha_venta,
                (v.precio_venta - p.precio_compra) * v.cantidad as beneficio,
                ((v.precio_venta - p.precio_compra) / p.precio_compra * 100) as margen_porcentaje
            FROM ventas v
            JOIN productos p ON v.producto_id = p.id
            WHERE (:start_date IS NULL OR v.fecha_venta >= :start_date)
            AND (:end_date IS NULL OR v.fecha_venta <= :end_date)
            ORDER BY v.fecha_venta DESC
        """)
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={'start_date': start_date, 'end_date': end_date})
            return df

    def get_inventory_report(self):
        """Obtiene reporte de inventario actual"""
        query = text("""
            SELECT 
                p.sku,
                p.nombre,
                p.categoria,
                p.stock_actual,
                p.stock_minimo,
                p.precio_compra,
                p.precio_venta,
                p.stock_actual * p.precio_compra as valor_inventario,
                COALESCE(SUM(v.cantidad), 0) as ventas_totales,
                COALESCE(AVG(v.precio_venta), 0) as precio_promedio_venta
            FROM productos p
            LEFT JOIN ventas v ON p.id = v.producto_id
            GROUP BY p.id
            ORDER BY p.categoria, p.nombre
        """)
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)
            return df

    def get_performance_report(self, periodo='month'):
        """Obtiene reporte de rendimiento por periodo"""
        query = text("""
            SELECT 
                DATE_TRUNC(:periodo, v.fecha_venta) as periodo,
                COUNT(DISTINCT v.producto_id) as productos_vendidos,
                SUM(v.cantidad) as unidades_vendidas,
                SUM(v.precio_venta * v.cantidad) as ingresos_totales,
                SUM((v.precio_venta - p.precio_compra) * v.cantidad) as beneficio_total
            FROM ventas v
            JOIN productos p ON v.producto_id = p.id
            GROUP BY DATE_TRUNC(:periodo, v.fecha_venta)
            ORDER BY periodo DESC
        """)
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={'periodo': periodo})
            return df