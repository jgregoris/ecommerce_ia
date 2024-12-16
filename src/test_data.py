import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from database import DatabaseManager
from sqlalchemy import text
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def insert_test_products(db):
    """Inserta productos de prueba en la base de datos"""
    productos = [
        {
            "sku": "AUD-BT001",
            "nombre": "Auriculares Bluetooth",
            "precio_compra": 25.00,
            "precio_venta": 49.99,
            "stock_actual": 50,
            "stock_minimo": 10,
            "categoria": "Electrónica"
        },
        {
            "sku": "FUND-MOV001",
            "nombre": "Funda Móvil Universal",
            "precio_compra": 3.50,
            "precio_venta": 9.99,
            "stock_actual": 100,
            "stock_minimo": 20,
            "categoria": "Accesorios"
        },
        {
            "sku": "CARG-USB001",
            "nombre": "Cargador USB-C Rápido",
            "precio_compra": 8.00,
            "precio_venta": 19.99,
            "stock_actual": 75,
            "stock_minimo": 15,
            "categoria": "Electrónica"
        },
        {
            "sku": "SPK-BT001",
            "nombre": "Altavoz Bluetooth Portátil",
            "precio_compra": 15.00,
            "precio_venta": 39.99,
            "stock_actual": 30,
            "stock_minimo": 8,
            "categoria": "Electrónica"
        },
        {
            "sku": "PWB-10000",
            "nombre": "Powerbank 10000mAh",
            "precio_compra": 12.00,
            "precio_venta": 29.99,
            "stock_actual": 45,
            "stock_minimo": 10,
            "categoria": "Electrónica"
        },
        {
            "sku": "PROT-SCR001",
            "nombre": "Protector Pantalla Templado",
            "precio_compra": 1.50,
            "precio_venta": 7.99,
            "stock_actual": 200,
            "stock_minimo": 50,
            "categoria": "Accesorios"
        },
        {
            "sku": "SOP-COCHE001",
            "nombre": "Soporte Móvil Coche",
            "precio_compra": 4.00,
            "precio_venta": 14.99,
            "stock_actual": 60,
            "stock_minimo": 15,
            "categoria": "Accesorios"
        },
        {
            "sku": "CAM-WEB001",
            "nombre": "Cámara Web HD",
            "precio_compra": 18.00,
            "precio_venta": 45.99,
            "stock_actual": 25,
            "stock_minimo": 5,
            "categoria": "Electrónica"
        },
        {
            "sku": "MOU-GAME001",
            "nombre": "Ratón Gaming RGB",
            "precio_compra": 10.00,
            "precio_venta": 29.99,
            "stock_actual": 40,
            "stock_minimo": 8,
            "categoria": "Gaming"
        },
        {
            "sku": "TEC-MEC001",
            "nombre": "Teclado Mecánico RGB",
            "precio_compra": 35.00,
            "precio_venta": 79.99,
            "stock_actual": 20,
            "stock_minimo": 5,
            "categoria": "Gaming"
        }
    ]
    
    for producto in productos:
        try:
            db.add_product(
                producto["sku"],
                producto["nombre"],
                producto["precio_compra"],
                producto["precio_venta"],
                producto["stock_actual"],
                producto["stock_minimo"],
                producto["categoria"]
            )
            logger.info(f"Producto añadido: {producto['nombre']}")
        except Exception as e:
            logger.error(f"Error al añadir {producto['nombre']}: {str(e)}")

def insert_test_sales(db):
    """Genera e inserta ventas de prueba"""
    try:
        # Obtener IDs de productos
        productos = pd.read_sql("SELECT id, precio_venta FROM productos", db.engine)
        
        if productos.empty:
            logger.error("No hay productos disponibles para generar ventas")
            return
        
        # Generar ventas aleatorias para los últimos 30 días
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        ventas_generadas = 0
        
        for date in dates:
            # Generar 2-5 ventas por día
            n_sales = np.random.randint(2, 6)
            for _ in range(n_sales):
                # Seleccionar producto aleatorio
                producto = productos.sample(1).iloc[0]
                
                # Generar cantidad aleatoria (1-5 unidades)
                cantidad = np.random.randint(1, 5)
                
                # Variación aleatoria en el precio (±10%)
                precio_base = float(producto['precio_venta'])
                precio_venta = round(precio_base * np.random.uniform(0.9, 1.1), 2)
                
                try:
                    db.register_sale(
                        producto_id=producto['id'],
                        cantidad=cantidad,
                        precio_venta=precio_venta
                    )
                    ventas_generadas += 1
                    if ventas_generadas % 10 == 0:  # Log cada 10 ventas
                        logger.info(f"Generadas {ventas_generadas} ventas...")
                except Exception as e:
                    logger.error(f"Error al registrar venta para producto {producto['id']}: {str(e)}")
        
        logger.info(f"Proceso completado. Total de ventas generadas: {ventas_generadas}")
        
    except Exception as e:
        logger.error(f"Error en la generación de ventas: {str(e)}")
        
def generate_test_data():
    """Función principal para generar todos los datos de prueba"""
    try:
        # Inicializar conexión a base de datos
        db = DatabaseManager()
        
        # Verificar conexión
        logger.info("Verificando conexión a la base de datos...")
        try:
            with db.engine.connect() as conn:
                conn.execute(text("SELECT 1"))  # Usamos text() aquí
            logger.info("Conexión exitosa")
        except Exception as e:
            logger.error(f"Error de conexión: {e}")
            return
        
        # Insertar productos
        logger.info("Iniciando inserción de productos...")
        insert_test_products(db)
        logger.info("Productos insertados correctamente")
        
        # Generar ventas
        logger.info("Iniciando generación de ventas...")
        insert_test_sales(db)
        logger.info("Ventas generadas correctamente")
        
        logger.info("Proceso de generación de datos completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error en la generación de datos de prueba: {str(e)}")

if __name__ == "__main__":
    # Ejecutar generación de datos
    generate_test_data()