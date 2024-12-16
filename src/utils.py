import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Clase para manejar la configuración de la aplicación"""
    
    def __init__(self):
        self.DEFAULT_CONFIG = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "amazon_analytics",
                "user": "postgres",
                "password": ""
            },
            "app": {
                "theme": "plotly_white",
                "default_date_range": 30,
                "cache_timeout": 3600,
                "debug": False
            },
            "prediction": {
                "forecast_days": 90,
                "confidence_interval": 0.95,
                "min_historical_days": 30
            },
            "clustering": {
                "n_clusters": 5,
                "min_samples": 10
            }
        }
        self.config = self.load_config()

    def load_config(self):
        """Carga la configuración desde archivo"""
        try:
            config_path = Path("config.json")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return {**self.DEFAULT_CONFIG, **json.load(f)}
            return self.DEFAULT_CONFIG
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self.DEFAULT_CONFIG

    def save_config(self):
        """Guarda la configuración actual"""
        try:
            with open("config.json", 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving config: {e}")

class MetricsCalculator:
    """Clase para cálculos de métricas comunes"""
    
    @staticmethod
    def calculate_moving_average(data, window=7):
        """Calcula la media móvil de una serie"""
        return data.rolling(window=window, min_periods=1).mean()

    @staticmethod
    def calculate_growth_rate(data):
        """Calcula la tasa de crecimiento"""
        return (data - data.shift(1)) / data.shift(1) * 100

    @staticmethod
    def calculate_stock_turnover(sales, stock):
        """Calcula la rotación de inventario"""
        return np.where(stock > 0, sales / stock, 0)

    @staticmethod
    def calculate_margin(revenue, cost):
        """Calcula el margen de beneficio"""
        return np.where(revenue > 0, (revenue - cost) / revenue * 100, 0)

class DataValidator:
    """Clase para validación de datos"""
    
    @staticmethod
    def validate_product_data(data):
        """Valida datos de productos"""
        required_columns = ['nombre', 'precio_compra', 'precio_venta', 'stock_actual', 'stock_minimo', 'categoria']
        
        # Verificar columnas requeridas
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            raise ValueError(f"Faltan columnas requeridas: {missing}")
            
        # Verificar tipos de datos
        if not all(pd.to_numeric(data[col], errors='coerce').notna().all() 
                  for col in ['precio_compra', 'precio_venta', 'stock_actual', 'stock_minimo']):
            raise ValueError("Valores numéricos inválidos en columnas de precio o stock")
            
        # Verificar valores negativos
        if (data[['precio_compra', 'precio_venta', 'stock_actual', 'stock_minimo']] < 0).any().any():
            raise ValueError("Se encontraron valores negativos en precio o stock")
            
        return True

    @staticmethod
    def validate_sales_data(data):
        """Valida datos de ventas"""
        required_columns = ['producto_id', 'cantidad', 'precio_venta', 'fecha_venta']
        
        # Verificar columnas requeridas
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            raise ValueError(f"Faltan columnas requeridas: {missing}")
            
        # Verificar tipos de datos
        if not pd.to_datetime(data['fecha_venta'], errors='coerce').notna().all():
            raise ValueError("Fechas inválidas en columna fecha_venta")
            
        if not all(pd.to_numeric(data[col], errors='coerce').notna().all() 
                  for col in ['cantidad', 'precio_venta']):
            raise ValueError("Valores numéricos inválidos en cantidad o precio_venta")
            
        # Verificar valores negativos
        if (data[['cantidad', 'precio_venta']] < 0).any().any():
            raise ValueError("Se encontraron valores negativos en cantidad o precio")
            
        return True

class DateHelper:
    """Clase para manejo de fechas"""
    
    @staticmethod
    def get_date_range(start_date=None, end_date=None, days=30):
        """Obtiene rango de fechas con valores por defecto"""
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=days)
            
        return start_date, end_date

    @staticmethod
    def generate_date_series(start_date, end_date, freq='D'):
        """Genera serie de fechas"""
        return pd.date_range(start=start_date, end=end_date, freq=freq)

    @staticmethod
    def get_period_dates(period='month'):
        """Obtiene fechas para diferentes períodos"""
        end_date = datetime.now()
        
        periods = {
            'day': 1,
            'week': 7,
            'month': 30,
            'quarter': 90,
            'year': 365
        }
        
        days = periods.get(period, 30)
        start_date = end_date - timedelta(days=days)
        
        return start_date, end_date

def format_currency(value):
    """Formatea valores monetarios"""
    return f"${value:,.2f}"

def format_percentage(value):
    """Formatea porcentajes"""
    return f"{value:.1f}%"

def format_number(value):
    """Formatea números enteros"""
    return f"{int(value):,}"
