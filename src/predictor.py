from prophet import Prophet
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SalesPredictor:
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
            future = model.make_future_dataframe(periods=90)
            
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
            print(f"Error en predict_sales: {str(e)}")
            # Retornar datos simulados en caso de error
            dates = pd.date_range(start=datetime.now(), periods=90, freq='D')
            return pd.DataFrame({
                'ds': dates,
                'yhat': np.zeros(len(dates)),
                'yhat_lower': np.zeros(len(dates)),
                'yhat_upper': np.zeros(len(dates))
            })

    def calculate_metrics(self, forecast, historical_data):
        """
        Calcula métricas adicionales para análisis
        """
        try:
            metrics = {}
            
            # Predicciones mensuales
            future_months = forecast[forecast['ds'] > datetime.now()]
            monthly_pred = future_months.set_index('ds').resample('M')[['yhat', 'yhat_lower', 'yhat_upper']].mean()
            
            # Tendencia
            trend_start = forecast['yhat'].iloc[0]
            trend_end = forecast['yhat'].iloc[-1]
            trend_change = ((trend_end - trend_start) / (trend_start + 1e-6)) * 100
            
            # Patrones semanales
            weekly_pattern = future_months.set_index('ds')
            weekly_pattern['weekday'] = weekly_pattern.index.dayofweek
            weekly_avg = weekly_pattern.groupby('weekday')['yhat'].mean()
            
            # Recomendaciones de stock
            stock_min = max(0, future_months['yhat_lower'].mean() * 7)  # 1 semana de stock mínimo
            stock_max = max(0, future_months['yhat_upper'].mean() * 14)  # 2 semanas de stock máximo
            
            metrics['monthly_predictions'] = monthly_pred
            metrics['trend_change'] = trend_change
            metrics['weekly_pattern'] = weekly_avg
            metrics['stock_recommendations'] = {
                'min_stock': round(stock_min, 2),
                'max_stock': round(stock_max, 2),
                'optimal_stock': round((stock_min + stock_max) / 2, 2)
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error en calculate_metrics: {str(e)}")
            return {
                'monthly_predictions': pd.DataFrame(),
                'trend_change': 0,
                'weekly_pattern': pd.Series(),
                'stock_recommendations': {
                    'min_stock': 0,
                    'max_stock': 0,
                    'optimal_stock': 0
                }
            }