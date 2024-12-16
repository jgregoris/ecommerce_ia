import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import numpy as np
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        """Inicializa la conexión a la base de datos"""
        self.DATABASE_URL = "postgresql://localhost:5432/amazon_analytics"
        try:
            self.engine = create_engine(self.DATABASE_URL)
            logger.info("Conexión a base de datos establecida")
        except Exception as e:
            logger.error(f"Error al conectar a la base de datos: {e}")
            raise

    def test_connection(self):
        """Prueba la conexión a la base de datos"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Error en test_connection: {e}")
            return False

    def add_product(self, sku, nombre, precio_compra, precio_venta, stock_actual, stock_minimo, categoria):
        """Añade un nuevo producto a la base de datos"""
        try:
            query = text("""
                INSERT INTO productos (sku, nombre, precio_compra, precio_venta, stock_actual, stock_minimo, categoria)
                VALUES (:sku, :nombre, :precio_compra, :precio_venta, :stock_actual, :stock_minimo, :categoria)
                RETURNING id
            """)
            with self.engine.begin() as conn:
                result = conn.execute(query, {
                    'sku': sku,
                    'nombre': nombre,
                    'precio_compra': precio_compra,
                    'precio_venta': precio_venta,
                    'stock_actual': stock_actual,
                    'stock_minimo': stock_minimo,
                    'categoria': categoria
                })
                return result.scalar()
        except Exception as e:
            logger.error(f"Error en add_product: {e}")
            raise Exception(f"Error al añadir producto: {str(e)}")

    def update_product(self, product_id, sku, nombre, precio_compra, precio_venta, stock_actual, stock_minimo, categoria):
        """Actualiza un producto existente"""
        try:
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
            
            with self.engine.begin() as conn:
                result = conn.execute(query, {
                    'product_id': product_id,
                    'sku': sku,
                    'nombre': nombre,
                    'precio_compra': precio_compra,
                    'precio_venta': precio_venta,
                    'stock_actual': stock_actual,
                    'stock_minimo': stock_minimo,
                    'categoria': categoria
                })
                if result.rowcount == 0:
                    raise ValueError("Producto no encontrado")
                return True
        except Exception as e:
            logger.error(f"Error en update_product: {e}")
            raise Exception(f"Error al actualizar producto: {str(e)}")
        
        """Actualiza un producto existente"""
        try:
            query = text("""
                UPDATE productos 
                SET sku = :sku,
                    nombre = :nombre,
                    precio_compra = :precio_compra,
                    precio_venta = :precio_venta,
                    stock_actual = :stock_actual,
                    stock_minimo = :stock_minimo,
                    categoria = :categoria,
                    fecha_actualizacion = CURRENT_TIMESTAMP
                WHERE id = :product_id
            """)
            
            with self.engine.begin() as conn:
                result = conn.execute(query, {
                    'product_id': product_id,
                    'sku': sku,
                    'nombre': nombre,
                    'precio_compra': precio_compra,
                    'precio_venta': precio_venta,
                    'stock_actual': stock_actual,
                    'stock_minimo': stock_minimo,
                    'categoria': categoria
                })
                if result.rowcount == 0:
                    raise ValueError("Producto no encontrado")
                return True
        except Exception as e:
            logger.error(f"Error en update_product: {e}")
            raise Exception(f"Error al actualizar producto: {str(e)}")

    def register_sale(self, producto_id, cantidad, precio_venta):
        """Registra una venta y actualiza el stock"""
        try:
            with self.engine.begin() as conn:
                # Verificar stock disponible
                query_stock = text("""
                    SELECT stock_actual 
                    FROM productos 
                    WHERE id = :producto_id
                    FOR UPDATE
                """)
                result = conn.execute(query_stock, {"producto_id": producto_id})
                stock_actual = result.scalar()
                
                if stock_actual is None:
                    raise ValueError("Producto no encontrado")
                    
                if stock_actual < cantidad:
                    raise ValueError(f"Stock insuficiente. Stock actual: {stock_actual}")
                
                # Registrar venta
                query_venta = text("""
                    INSERT INTO ventas (producto_id, cantidad, precio_venta, fecha_venta)
                    VALUES (:producto_id, :cantidad, :precio_venta, CURRENT_TIMESTAMP)
                    RETURNING id
                """)
                result = conn.execute(
                    query_venta,
                    {
                        'producto_id': producto_id,
                        'cantidad': cantidad,
                        'precio_venta': precio_venta
                    }
                )
                venta_id = result.scalar()
                
                # Actualizar stock (quitamos fecha_actualizacion)
                query_update = text("""
                    UPDATE productos 
                    SET stock_actual = stock_actual - :cantidad
                    WHERE id = :producto_id
                """)
                conn.execute(
                    query_update,
                    {
                        'producto_id': producto_id,
                        'cantidad': cantidad
                    }
                )
                
                return venta_id
                    
        except ValueError as e:
            logger.warning(f"Error de validación en register_sale: {e}")
            raise
        except Exception as e:
            logger.error(f"Error en register_sale: {e}")
            raise Exception(f"Error al registrar la venta: {str(e)}")
    
        """Registra una venta y actualiza el stock"""
        try:
            with self.engine.begin() as conn:
                # Verificar stock disponible
                query_stock = text("""
                    SELECT stock_actual 
                    FROM productos 
                    WHERE id = :producto_id
                    FOR UPDATE
                """)
                result = conn.execute(query_stock, {"producto_id": producto_id})
                stock_actual = result.scalar()
                
                if stock_actual is None:
                    raise ValueError("Producto no encontrado")
                    
                if stock_actual < cantidad:
                    raise ValueError(f"Stock insuficiente. Stock actual: {stock_actual}")
                
                # Registrar venta
                query_venta = text("""
                    INSERT INTO ventas (producto_id, cantidad, precio_venta, fecha_venta)
                    VALUES (:producto_id, :cantidad, :precio_venta, CURRENT_TIMESTAMP)
                    RETURNING id
                """)
                result = conn.execute(
                    query_venta,
                    {
                        'producto_id': producto_id,
                        'cantidad': cantidad,
                        'precio_venta': precio_venta
                    }
                )
                venta_id = result.scalar()
                
                # Actualizar stock
                query_update = text("""
                    UPDATE productos 
                    SET stock_actual = stock_actual - :cantidad,
                        fecha_actualizacion = CURRENT_TIMESTAMP
                    WHERE id = :producto_id
                """)
                conn.execute(
                    query_update,
                    {
                        'producto_id': producto_id,
                        'cantidad': cantidad
                    }
                )
                
                return venta_id
                
        except ValueError as e:
            logger.warning(f"Error de validación en register_sale: {e}")
            raise
        except Exception as e:
            logger.error(f"Error en register_sale: {e}")
            raise Exception(f"Error al registrar la venta: {str(e)}")

    def get_product_sales(self, producto_id):
        """Obtiene el historial de ventas de un producto con formato para Prophet"""
        try:
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
            
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params={'producto_id': producto_id})
                if df.empty:
                    dates = pd.date_range(
                        start=datetime.now() - timedelta(days=90),
                        end=datetime.now(),
                        freq='D'
                    )
                    df = pd.DataFrame({
                        'ds': dates,
                        'y': np.random.normal(10, 2, size=len(dates))
                    })
                    df['y'] = df['y'].clip(lower=0)
                return df
        except Exception as e:
            logger.error(f"Error en get_product_sales: {e}")
            return None

    def get_alerts(self):
        """Obtiene todas las alertas activas"""
        alerts = {
            'critical': [],
            'warning': [],
            'opportunity': []
        }
        
        try:
            stock_alerts = pd.read_sql(text("""
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
            """), self.engine)
            
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
            logger.error(f"Error en get_alerts: {e}")
            return alerts

    def load_dashboard_data(self, fecha_inicio=None, fecha_fin=None):
        """Carga los datos para el dashboard principal"""
        try:
            logger.info(f"Cargando datos del dashboard para el período: {fecha_inicio} - {fecha_fin}")
            
            # Si no se especifican fechas, usar últimos 30 días
            if fecha_inicio is None:
                fecha_inicio = datetime.now().date() - timedelta(days=30)
            if fecha_fin is None:
                fecha_fin = datetime.now().date()
                
            # Query para métricas principales con valores por defecto cuando no hay ventas
            metrics = pd.read_sql(text("""
                WITH base_metrics AS (
                    SELECT 
                        COALESCE(SUM(v.cantidad * v.precio_venta), 0) as total_ventas,
                        COALESCE(COUNT(DISTINCT v.producto_id), 0) as productos_vendidos,
                        COALESCE(SUM(v.cantidad), 0) as unidades_vendidas,
                        COALESCE(AVG(v.precio_venta), 0) as ticket_promedio,
                        COALESCE(AVG((v.precio_venta - p.precio_compra) / NULLIF(v.precio_venta, 0) * 100), 0) as margen_promedio,
                        COALESCE(SUM(v.cantidad * (v.precio_venta - p.precio_compra)), 0) as beneficio_total
                    FROM productos p
                    LEFT JOIN ventas v ON v.producto_id = p.id 
                        AND v.fecha_venta BETWEEN :fecha_inicio AND :fecha_fin
                ),
                productos_metrics AS (
                    SELECT 
                        COUNT(id) as productos_total,
                        SUM(stock_actual) as stock_total,
                        SUM(stock_actual * precio_compra) as valor_inventario,
                        COUNT(CASE WHEN stock_actual = 0 THEN 1 END) as productos_sin_stock,
                        COUNT(CASE WHEN stock_actual <= stock_minimo THEN 1 END) as productos_stock_bajo,
                        COUNT(CASE WHEN stock_actual > 0 THEN 1 END) as productos_activos,
                        COUNT(CASE WHEN stock_actual > stock_minimo * 2 THEN 1 END) as productos_alta_rotacion
                    FROM productos
                )
                SELECT 
                    bm.*,
                    pm.*,
                    COALESCE(
                        (SELECT SUM(v2.cantidad * v2.precio_venta)
                        FROM ventas v2
                        WHERE v2.fecha_venta BETWEEN 
                            :fecha_inicio - (:fecha_fin - :fecha_inicio)::interval 
                            AND :fecha_inicio), 0
                    ) as ventas_periodo_anterior
                FROM base_metrics bm
                CROSS JOIN productos_metrics pm
            """), self.engine, params={'fecha_inicio': fecha_inicio, 'fecha_fin': fecha_fin})

            # Datos de ventas diarias para gráficos
            ventas_diarias = pd.read_sql(text("""
                WITH fecha_serie AS (
                    SELECT generate_series(
                        :fecha_inicio::timestamp, 
                        :fecha_fin::timestamp, 
                        '1 day'::interval
                    )::date as fecha
                )
                SELECT 
                    fs.fecha,
                    COALESCE(SUM(v.cantidad * v.precio_venta), 0) as venta_total
                FROM fecha_serie fs
                LEFT JOIN ventas v ON DATE(v.fecha_venta) = fs.fecha
                GROUP BY fs.fecha
                ORDER BY fs.fecha
            """), self.engine, params={'fecha_inicio': fecha_inicio, 'fecha_fin': fecha_fin})

            if not ventas_diarias.empty:
                # Calcular media móvil de 7 días
                ventas_diarias['media_movil'] = ventas_diarias['venta_total'].rolling(window=7, min_periods=1).mean()
                # Añadir a las métricas
                metrics['fecha_ventas'] = ventas_diarias['fecha'].tolist()
                metrics['ventas_diarias'] = ventas_diarias['venta_total'].tolist()
                metrics['media_movil'] = ventas_diarias['media_movil'].tolist()

            # Query para productos con stock bajo
            low_stock = pd.read_sql(text("""
                SELECT 
                    id, sku, nombre, stock_actual, stock_minimo,
                    CAST((stock_actual::float / NULLIF(stock_minimo, 0) * 100) AS DECIMAL(10,2)) as stock_percentage
                FROM productos
                WHERE stock_actual <= stock_minimo * 1.2
                ORDER BY (stock_actual::float / NULLIF(stock_minimo, 0)) ASC
            """), self.engine)

            # Query para ventas recientes
            recent_sales = pd.read_sql(text("""
                SELECT 
                    p.nombre, 
                    v.cantidad, 
                    v.precio_venta,
                    CAST(v.precio_venta * v.cantidad AS DECIMAL(10,2)) as total_venta,
                    v.fecha_venta,
                    p.categoria
                FROM ventas v
                JOIN productos p ON v.producto_id = p.id
                WHERE v.fecha_venta BETWEEN :fecha_inicio AND :fecha_fin
                ORDER BY v.fecha_venta DESC
                LIMIT 10
            """), self.engine, params={'fecha_inicio': fecha_inicio, 'fecha_fin': fecha_fin})

            logger.info("Datos del dashboard cargados exitosamente")
            return low_stock, recent_sales, metrics

        except Exception as e:
            logger.error(f"Error en load_dashboard_data: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    def get_inventory_report(self):
        """Obtiene reporte de inventario actual"""
        try:
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
            
            with self.engine.connect() as conn:
                return pd.read_sql(query, conn)
        except Exception as e:
            logger.error(f"Error en get_inventory_report: {e}")
            return pd.DataFrame()

    def get_sales_report(self, start_date=None, end_date=None):
        """Obtiene reporte detallado de ventas"""
        try:
            query = text("""
                SELECT 
                    p.nombre as producto,
                    p.categoria,
                    v.cantidad,
                    v.precio_venta,
                    v.fecha_venta,
                    CAST(v.precio_venta * v.cantidad AS DECIMAL(10,2)) as total_venta,
                    CAST((v.precio_venta - p.precio_compra) * v.cantidad AS DECIMAL(10,2)) as beneficio,
                    CAST(((v.precio_venta - p.precio_compra) / NULLIF(v.precio_venta, 0) * 100) AS DECIMAL(10,2)) as margen_porcentaje
                FROM ventas v
                JOIN productos p ON v.producto_id = p.id
                WHERE (:start_date IS NULL OR v.fecha_venta >= :start_date)
                AND (:end_date IS NULL OR v.fecha_venta <= :end_date)
                ORDER BY v.fecha_venta DESC
            """)
            
            with self.engine.connect() as conn:
                return pd.read_sql(query, conn, params={'start_date': start_date, 'end_date': end_date})
        except Exception as e:
            logger.error(f"Error en get_sales_report: {e}")
            return pd.DataFrame()

    def get_performance_report(self, periodo='month'):
        """Obtiene reporte de rendimiento por periodo"""
        try:
            # Adaptamos el intervalo según el período seleccionado
            interval_map = {
                'day': "1 day",
                'week': "1 week",
                'month': "1 month"
            }
            
            query = text("""
                WITH ventas_periodo AS (
                    SELECT 
                        DATE_TRUNC(:periodo, v.fecha_venta) as periodo,
                        p.categoria,
                        COUNT(DISTINCT v.producto_id) as productos_vendidos,
                        SUM(v.cantidad) as unidades_vendidas,
                        CAST(SUM(v.cantidad * v.precio_venta) AS DECIMAL(10,2)) as ingresos_totales,
                        CAST(SUM(v.cantidad * (v.precio_venta - p.precio_compra)) AS DECIMAL(10,2)) as beneficio_total,
                        CAST(
                            (SUM(v.cantidad * (v.precio_venta - p.precio_compra)) / 
                            NULLIF(SUM(v.cantidad * v.precio_venta), 0) * 100) 
                        AS DECIMAL(10,2)) as margen_porcentaje
                    FROM ventas v
                    JOIN productos p ON v.producto_id = p.id
                    WHERE v.fecha_venta >= CURRENT_DATE - INTERVAL '12 months'
                    GROUP BY DATE_TRUNC(:periodo, v.fecha_venta), p.categoria
                )
                SELECT 
                    periodo,
                    categoria,
                    productos_vendidos,
                    unidades_vendidas,
                    ingresos_totales,
                    beneficio_total,
                    margen_porcentaje
                FROM ventas_periodo
                ORDER BY periodo DESC, categoria
            """)
            
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params={'periodo': periodo})
                
                # Convertimos la columna periodo a datetime si no lo está
                df['periodo'] = pd.to_datetime(df['periodo'])
                
                return df
                
        except Exception as e:
            logger.error(f"Error en get_performance_report: {e}")
            return pd.DataFrame()
        
    def save_prediction(self, producto_id, prediccion):
        """Guarda una predicción en la base de datos"""
        try:
            query = text("""
                INSERT INTO predicciones 
                    (producto_id, valor_prediccion, intervalo_inferior, 
                    intervalo_superior, periodo_prediccion, confianza)
                VALUES 
                    (:producto_id, :valor_prediccion, :intervalo_inferior,
                    :intervalo_superior, :periodo_prediccion, :confianza)
                RETURNING id
            """)
            
            with self.engine.begin() as conn:
                result = conn.execute(query, {
                    'producto_id': producto_id,
                    'valor_prediccion': prediccion['yhat'].mean(),
                    'intervalo_inferior': prediccion['yhat_lower'].mean(),
                    'intervalo_superior': prediccion['yhat_upper'].mean(),
                    'periodo_prediccion': '30_dias',
                    'confianza': 0.95
                })
                return result.scalar()
        except Exception as e:
            logger.error(f"Error en save_prediction: {e}")
            raise

    def save_metrics(self, producto_id, metricas):
        """Guarda las métricas de un producto"""
        try:
            query = text("""
                INSERT INTO metricas 
                    (producto_id, rotacion_stock, margen_promedio, 
                    rentabilidad, tendencia_ventas, stock_optimo, dias_stock)
                VALUES 
                    (:producto_id, :rotacion_stock, :margen_promedio,
                    :rentabilidad, :tendencia_ventas, :stock_optimo, :dias_stock)
                RETURNING id
            """)
            
            with self.engine.begin() as conn:
                result = conn.execute(query, {
                    'producto_id': producto_id,
                    'rotacion_stock': metricas.get('rotacion', 0),
                    'margen_promedio': metricas.get('margen', 0),
                    'rentabilidad': metricas.get('rentabilidad', 0),
                    'tendencia_ventas': metricas.get('tendencia', 0),
                    'stock_optimo': metricas.get('stock_optimo', 0),
                    'dias_stock': metricas.get('dias_stock', 0)
                })
                return result.scalar()
        except Exception as e:
            logger.error(f"Error en save_metrics: {e}")
            raise

    def save_clustering(self, producto_id, cluster_data):
        """Guarda los resultados del clustering"""
        try:
            query = text("""
                INSERT INTO clustering 
                    (producto_id, cluster_id, similitud_score, 
                    caracteristicas, descripcion)
                VALUES 
                    (:producto_id, :cluster_id, :similitud_score,
                    :caracteristicas, :descripcion)
                RETURNING id
            """)
            
            with self.engine.begin() as conn:
                result = conn.execute(query, {
                    'producto_id': producto_id,
                    'cluster_id': cluster_data['cluster_id'],
                    'similitud_score': cluster_data['similitud'],
                    'caracteristicas': cluster_data['caracteristicas'],
                    'descripcion': cluster_data['descripcion']
                })
                return result.scalar()
        except Exception as e:
            logger.error(f"Error en save_clustering: {e}")
            raise

    def save_anomaly(self, producto_id, anomalia_data):
        """Guarda una anomalía detectada"""
        try:
            query = text("""
                INSERT INTO anomalias 
                    (producto_id, tipo_anomalia, valor_detectado,
                    valor_esperado, score_anomalia, descripcion)
                VALUES 
                    (:producto_id, :tipo_anomalia, :valor_detectado,
                    :valor_esperado, :score_anomalia, :descripcion)
                RETURNING id
            """)
            
            with self.engine.begin() as conn:
                result = conn.execute(query, {
                    'producto_id': producto_id,
                    'tipo_anomalia': anomalia_data['tipo'],
                    'valor_detectado': anomalia_data['valor_detectado'],
                    'valor_esperado': anomalia_data['valor_esperado'],
                    'score_anomalia': anomalia_data['score'],
                    'descripcion': anomalia_data['descripcion']
                })
                return result.scalar()
        except Exception as e:
            logger.error(f"Error en save_anomaly: {e}")
            raise

    def get_latest_predictions(self, producto_id):
        """Obtiene las últimas predicciones de un producto"""
        try:
            query = text("""
                SELECT *
                FROM predicciones
                WHERE producto_id = :producto_id
                ORDER BY fecha_prediccion DESC
                LIMIT 1
            """)
            
            with self.engine.connect() as conn:
                result = pd.read_sql(query, conn, params={'producto_id': producto_id})
                return result.to_dict('records')[0] if not result.empty else None
        except Exception as e:
            logger.error(f"Error en get_latest_predictions: {e}")
            return None