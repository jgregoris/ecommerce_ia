from prophet import Prophet
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SalesPredictor:
    def __init__(self):
        """Inicializa el predictor"""
        pass

    def predict_sales(self, historical_data):
        """
        Predice ventas futuras asegurando valores no negativos
        """
        try:
            # Crear nueva instancia de Prophet para cada predicción
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.005,
                seasonality_prior_scale=0.1,
                interval_width=0.95
            )
            
            # Preparar datos
            historical_data['ds'] = pd.to_datetime(historical_data['ds'])
            historical_data['y'] = historical_data['y'].astype(float).clip(lower=0)
            
            # Entrenar modelo
            model.fit(historical_data)
            
            # Crear dataframe futuro
            future = model.make_future_dataframe(periods=90)  # 3 meses
            
            # Realizar predicción
            forecast = model.predict(future)
            
            # Asegurar valores no negativos y aplicar suavizado
            forecast['yhat'] = forecast['yhat'].clip(lower=0)
            forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
            forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
            
            # Suavizar predicciones
            forecast['yhat'] = forecast['yhat'].rolling(window=7, min_periods=1, center=True).mean()
            forecast['yhat_lower'] = forecast['yhat_lower'].rolling(window=7, min_periods=1, center=True).mean()
            forecast['yhat_upper'] = forecast['yhat_upper'].rolling(window=7, min_periods=1, center=True).mean()
            
            return forecast
        except Exception as e:
            logger.error(f"Error en predict_sales: {e}")
            return None

    def calculate_metrics(self, forecast, historical_data):
        """
        Calcula métricas para análisis y recomendaciones de stock
        """
        try:
            # Predicciones mensuales
            future_months = forecast[forecast['ds'] > datetime.now()]
            monthly_pred = future_months.set_index('ds').resample('M')[['yhat', 'yhat_lower', 'yhat_upper']].mean()
            
            # Calcular tendencia
            trend_start = forecast['yhat'].iloc[0]
            trend_end = forecast['yhat'].iloc[-1]
            trend_change = ((trend_end - trend_start) / (trend_start + 1e-6)) * 100
            
            # Patrones semanales
            weekly_pattern = future_months.set_index('ds')
            weekly_pattern['weekday'] = weekly_pattern.index.dayofweek
            weekly_avg = weekly_pattern.groupby('weekday')['yhat'].mean()
            
            # Calcular promedios y desviación estándar de ventas
            ventas_diarias = historical_data['y'].mean()
            ventas_std = historical_data['y'].std()

            # Cálculo de stock basado en ventas históricas
            if ventas_diarias > 0:
                # Stock mínimo: 3 días de ventas promedio + desviación estándar
                stock_min = round(ventas_diarias * 3 + ventas_std)
                
                # Stock óptimo: 7 días de ventas promedio
                stock_optimal = round(ventas_diarias * 7)
                
                # Stock máximo: 15 días de ventas promedio
                stock_max = round(ventas_diarias * 15)
                
                # Ajustar por tendencia
                if trend_change > 10:  # Tendencia alcista
                    stock_min = round(stock_min * 1.1)
                    stock_optimal = round(stock_optimal * 1.1)
                    stock_max = round(stock_max * 1.1)
                elif trend_change < -10:  # Tendencia bajista
                    stock_min = round(stock_min * 0.9)
                    stock_optimal = round(stock_optimal * 0.9)
                    stock_max = round(stock_max * 0.9)
                
                # Asegurar valores mínimos y orden correcto
                stock_min = max(1, stock_min)
                stock_optimal = max(stock_min + 1, stock_optimal)
                stock_max = max(stock_optimal + 1, stock_max)
                
                # Calcular días de stock
                dias_stock = round(stock_optimal / ventas_diarias) if ventas_diarias > 0 else 7
            else:
                # Valores por defecto si no hay ventas históricas
                stock_min = 1
                stock_optimal = 3
                stock_max = 5
                dias_stock = 7

            metrics = {
                'monthly_predictions': monthly_pred,
                'trend_change': trend_change,
                'weekly_pattern': weekly_avg,
                'stock_recommendations': {
                    'min_stock': stock_min,
                    'max_stock': stock_max,
                    'optimal_stock': stock_optimal
                },
                'rotacion': ventas_diarias,
                'margen': 0,
                'rentabilidad': 0,
                'tendencia': trend_change,
                'stock_optimo': stock_optimal,
                'dias_stock': dias_stock
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Error en calculate_metrics: {e}")
            return {
                'monthly_predictions': pd.DataFrame(),
                'trend_change': 0,
                'weekly_pattern': pd.Series(),
                'stock_recommendations': {
                    'min_stock': 1,
                    'max_stock': 5,
                    'optimal_stock': 3
                },
                'rotacion': 0,
                'margen': 0,
                'rentabilidad': 0,
                'tendencia': 0,
                'stock_optimo': 3,
                'dias_stock': 7
            }
