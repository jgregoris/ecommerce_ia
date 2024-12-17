import pandas as pd
from langchain.llms import Ollama
from typing import Dict, Optional
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SalesPredictor:
    def __init__(self):
        """Inicializa el predictor de ventas usando Llama 3.2"""
        try:
            self.llm = Ollama(model="llama2")
            logger.info("SalesPredictor inicializado con Llama 3.2")
        except Exception as e:
            logger.error(f"Error al inicializar SalesPredictor: {e}")
            raise

    def predict_sales(self, historical_data: pd.DataFrame, periods: int = 30) -> Optional[pd.DataFrame]:
        """
        Genera predicciones de ventas usando análisis estadístico
        """
        try:
            if historical_data is None or historical_data.empty:
                logger.error("No hay datos históricos disponibles")
                return None

            # Asegurar que las columnas existan y los datos sean numéricos
            historical_data['y'] = pd.to_numeric(historical_data['y'], errors='coerce')
            historical_data = historical_data.dropna(subset=['y'])

            # Calcular estadísticas básicas
            mean_sales = historical_data['y'].mean() or 0
            std_sales = historical_data['y'].std() or 0
            trend = self._calculate_trend(historical_data) or 0
            seasonality = self._calculate_seasonality(historical_data)

            # Generar fechas futuras
            last_date = pd.to_datetime(historical_data['ds'].max())
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=periods,
                freq='D'
            )

            # Generar predicciones base
            predictions = []
            lower_bounds = []
            upper_bounds = []

            for i, date in enumerate(future_dates):
                # Predicción base usando la media y tendencia
                base_prediction = max(0, mean_sales * (1 + trend * (i/30)))
                
                # Ajustar por estacionalidad
                weekday = date.weekday()
                seasonal_factor = seasonality.get(weekday, 1.0)
                
                # Añadir variación controlada
                prediction = max(0, base_prediction * seasonal_factor)
                
                # Calcular intervalos de confianza
                std_factor = 1.96  # 95% intervalo de confianza
                lower = max(0, prediction - std_factor * std_sales)
                upper = prediction + std_factor * std_sales

                predictions.append(prediction)
                lower_bounds.append(lower)
                upper_bounds.append(upper)

            # Crear DataFrame de predicciones
            forecast = pd.DataFrame({
                'ds': future_dates,
                'yhat': predictions,
                'yhat_lower': lower_bounds,
                'yhat_upper': upper_bounds
            })

            return forecast

        except Exception as e:
            logger.error(f"Error en predict_sales: {e}")
            return None

    def _calculate_trend(self, data: pd.DataFrame) -> float:
        """Calcula la tendencia en los datos históricos"""
        try:
            if len(data) < 2:
                return 0.0
                
            # Calcular tendencia como cambio porcentual promedio
            sales = data['y'].values
            diffs = np.diff(sales)
            avg_diff = np.mean(diffs)
            return avg_diff / (np.mean(sales[:-1]) + 1e-6)  # Evitar división por cero
        except Exception as e:
            logger.error(f"Error al calcular tendencia: {e}")
            return 0.0

    def _calculate_seasonality(self, data: pd.DataFrame) -> Dict[int, float]:
        """Calcula factores de estacionalidad por día de la semana"""
        try:
            # Convertir fechas a día de la semana
            data['weekday'] = pd.to_datetime(data['ds']).dt.weekday
            
            # Calcular factor estacional por día
            daily_avg = data.groupby('weekday')['y'].mean()
            overall_avg = data['y'].mean() or 1.0  # Evitar división por cero
            
            seasonality = {}
            for day in range(7):
                if day in daily_avg.index:
                    seasonality[day] = max(0.5, min(1.5, daily_avg[day] / overall_avg))
                else:
                    seasonality[day] = 1.0
                    
            return seasonality
        except Exception as e:
            logger.error(f"Error al calcular estacionalidad: {e}")
            return {i: 1.0 for i in range(7)}

    def calculate_metrics(self, forecast: pd.DataFrame, historical_data: pd.DataFrame) -> Dict:
        """Calcula métricas y recomendaciones basadas en las predicciones"""
        try:
            if forecast is None or historical_data is None:
                raise ValueError("Datos de predicción o históricos no disponibles")

            # Calcular tendencia con manejo de errores
            last_30_pred = forecast['yhat'].tail(30).mean() or 0
            prev_30_hist = historical_data['y'].tail(30).mean() or 0
            if prev_30_hist > 0:
                trend_change = ((last_30_pred - prev_30_hist) / prev_30_hist * 100)
            else:
                trend_change = 0

            # Predicciones mensuales
            monthly_pred = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)

            # Recomendaciones de stock con valores mínimos sensatos
            daily_avg = max(1, monthly_pred['yhat'].mean())
            stock_min = max(15, daily_avg * 15)
            stock_max = max(45, daily_avg * 45)
            stock_optimal = max(30, daily_avg * 30)

            # Calcular patrón semanal usando datos históricos
            weekly_pattern = pd.DataFrame()
            try:
                if not historical_data.empty:
                    # Convertir la columna ds a datetime si no lo es ya
                    historical_data['ds'] = pd.to_datetime(historical_data['ds'])
                    # Crear el día de la semana como string para mejor visualización
                    historical_data['weekday'] = historical_data['ds'].dt.strftime('%A')
                    # Calcular la media de ventas por día
                    weekly_pattern = historical_data.groupby('weekday')['y'].agg([
                        ('ventas_promedio', 'mean'),
                        ('conteo', 'count')
                    ]).reset_index()
                    # Ordenar los días de la semana correctamente
                    dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    weekly_pattern['weekday'] = pd.Categorical(weekly_pattern['weekday'], categories=dias_orden, ordered=True)
                    weekly_pattern = weekly_pattern.sort_values('weekday')
                    # Traducir los nombres de los días
                    dias_es = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
                    weekly_pattern['dia'] = dias_es
            except Exception as e:
                logger.error(f"Error al calcular patrón semanal: {e}")
                weekly_pattern = pd.DataFrame({
                    'dia': dias_es,
                    'ventas_promedio': [0] * 7,
                    'conteo': [0] * 7
                })

            return {
                'trend_change': float(trend_change),
                'monthly_predictions': monthly_pred,
                'stock_recommendations': {
                    'min_stock': float(stock_min),
                    'max_stock': float(stock_max),
                    'optimal_stock': float(stock_optimal)
                },
                'weekly_pattern': weekly_pattern
            }

        except Exception as e:
            logger.error(f"Error en calculate_metrics: {e}")
            return {
                'trend_change': 0.0,
                'monthly_predictions': pd.DataFrame(),
                'stock_recommendations': {
                    'min_stock': 15.0,
                    'max_stock': 45.0,
                    'optimal_stock': 30.0
                },
                'weekly_pattern': pd.DataFrame()
            }