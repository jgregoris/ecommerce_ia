from prophet import Prophet
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SalesPredictor:
    def __init__(self):
        # Configuramos el modelo con restricciones más estrictas
        self.model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,  # Desactivamos estacionalidad anual por tener pocos datos
            changepoint_prior_scale=0.005,  # Reducimos para suavizar cambios bruscos
            seasonality_prior_scale=0.1,    # Reducimos el impacto de la estacionalidad
            interval_width=0.95
        )
    
    def predict_sales(self, historical_data):
        """
        Predice ventas futuras asegurando valores no negativos
        """
        # Preparar datos
        historical_data['ds'] = pd.to_datetime(historical_data['ds'])
        historical_data['y'] = historical_data['y'].astype(float).clip(lower=0)
        
        # Calcular el promedio y la desviación estándar de las ventas
        mean_sales = historical_data['y'].mean()
        std_sales = historical_data['y'].std()
        
        # Entrenar modelo
        self.model.fit(historical_data)
        
        # Crear dataframe futuro
        future = self.model.make_future_dataframe(periods=90)
        
        # Realizar predicción
        forecast = self.model.predict(future)
        
        # Asegurar que no hay valores negativos y aplicar límites razonables
        forecast['yhat'] = forecast['yhat'].clip(lower=0)
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
        
        # Suavizar predicciones extremas
        forecast['yhat'] = forecast['yhat'].rolling(window=7, min_periods=1, center=True).mean()
        forecast['yhat_lower'] = forecast['yhat_lower'].rolling(window=7, min_periods=1, center=True).mean()
        forecast['yhat_upper'] = forecast['yhat_upper'].rolling(window=7, min_periods=1, center=True).mean()
        
        return forecast
    
    def calculate_metrics(self, forecast, historical_data):
        """
        Calcula métricas adicionales para análisis
        """
        metrics = {}
        
        # Predicciones mensuales (asegurando valores no negativos)
        future_months = forecast[forecast['ds'] > datetime.now()]
        monthly_pred = future_months.set_index('ds').resample('M')[['yhat', 'yhat_lower', 'yhat_upper']].mean()
        
        # Tendencia (usando solo valores positivos)
        trend_start = max(0, forecast['yhat'].iloc[0])
        trend_end = max(0, forecast['yhat'].iloc[-1])
        if trend_start > 0:
            trend_change = ((trend_end - trend_start) / trend_start) * 100
        else:
            trend_change = 0
        
        # Estacionalidad
        weekly_pattern = future_months.set_index('ds').resample('D')[['yhat']].mean()
        weekly_pattern['weekday'] = weekly_pattern.index.dayofweek
        weekly_avg = weekly_pattern.groupby('weekday')['yhat'].mean()
        
        # Stock recomendado (usando solo valores positivos)
        stock_min = max(0, future_months['yhat_lower'].mean() * 7)
        stock_max = max(0, future_months['yhat_upper'].mean() * 14)
        
        metrics['monthly_predictions'] = monthly_pred
        metrics['trend_change'] = trend_change
        metrics['weekly_pattern'] = weekly_avg
        metrics['stock_recommendations'] = {
            'min_stock': round(stock_min, 2),
            'max_stock': round(stock_max, 2),
            'optimal_stock': round((stock_min + stock_max) / 2, 2)
        }
        
        return metrics