import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class DashboardVisualizer:
    def __init__(self, theme="plotly_white"):
        self.theme = theme

    def create_main_dashboard(self, sales_data, inventory_data, predictions_data):
        """Crea el dashboard principal con múltiples gráficos"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Tendencia de Ventas y Predicción",
                    "Ventas por Categoría",
                    "Stock Actual por Categoría",
                    "Rentabilidad por Categoría"
                ),
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )

            # 1. Tendencia de Ventas y Predicción
            if not sales_data.empty:
                # Datos históricos
                fig.add_trace(
                    go.Scatter(
                        x=sales_data['fecha_venta'],
                        y=sales_data['cantidad'],
                        name="Ventas Reales",
                        line=dict(color="#1f77b4", width=2)
                    ),
                    row=1, col=1
                )

                if predictions_data is not None and not predictions_data.empty:
                    # Predicción
                    fig.add_trace(
                        go.Scatter(
                            x=predictions_data['ds'],
                            y=predictions_data['yhat'],
                            name="Predicción",
                            line=dict(color="#2ca02c", width=2, dash='dash')
                        ),
                        row=1, col=1
                    )

                    # Intervalo de confianza
                    fig.add_trace(
                        go.Scatter(
                            x=predictions_data['ds'].tolist() + predictions_data['ds'].tolist()[::-1],
                            y=predictions_data['yhat_upper'].tolist() + predictions_data['yhat_lower'].tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(0,176,246,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='Intervalo de Confianza',
                            showlegend=True
                        ),
                        row=1, col=1
                    )

            # 2. Ventas por Categoría
            if not sales_data.empty:
                ventas_categoria = sales_data.groupby('categoria')['cantidad'].sum().reset_index()
                fig.add_trace(
                    go.Bar(
                        x=ventas_categoria['categoria'],
                        y=ventas_categoria['cantidad'],
                        name="Ventas por Categoría",
                        marker_color="#1f77b4"
                    ),
                    row=1, col=2
                )

            # 3. Stock Actual por Categoría
            if not inventory_data.empty:
                fig.add_trace(
                    go.Bar(
                        x=inventory_data['categoria'],
                        y=inventory_data['stock_actual'],
                        name="Stock Actual",
                        marker_color="#2ca02c"
                    ),
                    row=2, col=1
                )

            # 4. Rentabilidad por Categoría
            if not sales_data.empty:
                rentabilidad = sales_data.groupby('categoria')['beneficio'].sum().reset_index()
                fig.add_trace(
                    go.Bar(
                        x=rentabilidad['categoria'],
                        y=rentabilidad['beneficio'],
                        name="Rentabilidad",
                        marker_color="#ff7f0e"
                    ),
                    row=2, col=2
                )

            # Actualizar layout
            fig.update_layout(
                height=800,
                showlegend=True,
                title_text="Dashboard de Ventas e Inventario",
                template=self.theme
            )

            return fig

        except Exception as e:
            logger.error(f"Error al crear dashboard: {str(e)}")
            return go.Figure()

        """Crea visualizaciones para el análisis de ventas"""
        try:
            figures = {}
            
            # 1. Tendencia temporal
            fig_trend = go.Figure()
            
            for categoria in sales_data['categoria'].unique():
                df_cat = sales_data[sales_data['categoria'] == categoria]
                fig_trend.add_trace(
                    go.Scatter(
                        x=df_cat['fecha_venta'],
                        y=df_cat['cantidad'],
                        name=categoria,
                        mode='lines'
                    )
                )
            
            fig_trend.update_layout(
                title="Tendencia de Ventas por Categoría",
                xaxis_title="Fecha",
                yaxis_title="Cantidad",
                template=self.theme
            )
            figures['trend'] = fig_trend

            # 2. Distribución de ventas
            fig_dist = go.Figure()
            fig_dist.add_trace(
                go.Box(
                    x=sales_data['categoria'],
                    y=sales_data['cantidad'],
                    name="Distribución de Ventas"
                )
            )
            fig_dist.update_layout(
                title="Distribución de Ventas por Categoría",
                xaxis_title="Categoría",
                yaxis_title="Cantidad",
                template=self.theme
            )
            figures['distribution'] = fig_dist

            # 3. Análisis temporal
            sales_data['mes'] = pd.to_datetime(sales_data['fecha_venta']).dt.month
            ventas_mes = sales_data.groupby(['mes', 'categoria'])['cantidad'].mean().reset_index()
            
            fig_time = go.Figure()
            for categoria in ventas_mes['categoria'].unique():
                df_cat = ventas_mes[ventas_mes['categoria'] == categoria]
                fig_time.add_trace(
                    go.Scatter(
                        x=df_cat['mes'],
                        y=df_cat['cantidad'],
                        name=categoria,
                        mode='lines+markers'
                    )
                )
            
            fig_time.update_layout(
                title="Patrón Temporal de Ventas",
                xaxis_title="Mes",
                yaxis_title="Cantidad Promedio",
                template=self.theme
            )
            figures['temporal'] = fig_time

            # 4. Análisis de rentabilidad
            fig_profit = go.Figure(data=[
                go.Scatter(
                    x=sales_data['cantidad'],
                    y=sales_data['beneficio'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=sales_data['margen_porcentaje'],
                        colorscale='RdYlBu',
                        showscale=True
                    ),
                    text=sales_data['categoria'],
                    name='Ventas vs Beneficio'
                )
            ])
            
            fig_profit.update_layout(
                title="Análisis de Rentabilidad",
                xaxis_title="Cantidad Vendida",
                yaxis_title="Beneficio",
                template=self.theme
            )
            figures['profit'] = fig_profit

            return figures

        except Exception as e:
            logger.error(f"Error al crear análisis de ventas: {str(e)}")
            return {'trend': go.Figure(), 'distribution': go.Figure(), 
                    'temporal': go.Figure(), 'profit': go.Figure()}
    def create_sales_analysis(self, sales_data):
        """Crea visualizaciones para el análisis de ventas"""
        try:
            figures = {}
            
            # 1. Tendencia temporal
            fig_trend = go.Figure()
            for categoria in sales_data['categoria'].unique():
                df_cat = sales_data[sales_data['categoria'] == categoria]
                fig_trend.add_trace(
                    go.Scatter(
                        x=df_cat['fecha_venta'],
                        y=df_cat['cantidad'],
                        name=categoria,
                        mode='lines+markers'
                    )
                )
            
            fig_trend.update_layout(
                title="Tendencia de Ventas por Categoría",
                xaxis_title="Fecha",
                yaxis_title="Cantidad",
                template=self.theme
            )
            figures['trend'] = fig_trend

            # 2. Distribución
            fig_dist = go.Figure()
            fig_dist.add_trace(
                go.Box(
                    x=sales_data['categoria'],
                    y=sales_data['cantidad'],
                    name="Distribución de Ventas"
                )
            )
            fig_dist.update_layout(
                title="Distribución de Ventas por Categoría",
                template=self.theme
            )
            figures['distribution'] = fig_dist

            # Solo crear el gráfico de estacionalidad si hay suficientes datos
            if len(sales_data) > 0:
                sales_data['mes'] = pd.to_datetime(sales_data['fecha_venta']).dt.month
                ventas_mes = sales_data.groupby(['mes', 'categoria'])['cantidad'].sum().reset_index()
                
                fig_season = go.Figure()
                for categoria in ventas_mes['categoria'].unique():
                    df_cat = ventas_mes[ventas_mes['categoria'] == categoria]
                    fig_season.add_trace(
                        go.Scatter(
                            x=df_cat['mes'],
                            y=df_cat['cantidad'],
                            name=categoria,
                            mode='lines+markers'
                        )
                    )
                
                fig_season.update_layout(
                    title="Ventas Mensuales por Categoría",
                    xaxis_title="Mes",
                    yaxis_title="Cantidad Total",
                    template=self.theme
                )
                figures['seasonality'] = fig_season
            else:
                figures['seasonality'] = go.Figure()

            return figures
        except Exception as e:
            logger.error(f"Error al crear análisis de ventas: {e}")
            return {
                'trend': go.Figure(),
                'distribution': go.Figure(),
                'seasonality': go.Figure()
            }
    def create_inventory_analysis(self, inventory_data):
        """Crea visualizaciones para el análisis de inventario"""
        try:
            figures = {}
            
            # 1. Distribución de stock
            fig_stock = go.Figure(data=[
                go.Bar(
                    x=inventory_data['categoria'],
                    y=inventory_data['stock_actual'],
                    name='Stock Actual'
                )
            ])
            fig_stock.update_layout(
                title="Distribución de Stock por Categoría",
                xaxis_title="Categoría",
                yaxis_title="Stock Actual",
                template=self.theme
            )
            figures['stock_dist'] = fig_stock

            # 2. Análisis de rotación
            if 'ventas_totales' in inventory_data.columns:
                inventory_data['rotacion'] = inventory_data['ventas_totales'] / inventory_data['stock_actual'].replace(0, 1)
                
                fig_rotation = go.Figure(data=[
                    go.Scatter(
                        x=inventory_data['stock_actual'],
                        y=inventory_data['rotacion'],
                        mode='markers',
                        text=inventory_data['nombre'],
                        marker=dict(
                            size=10,
                            color=inventory_data['valor_inventario'],
                            colorscale='Viridis',
                            showscale=True
                        ),
                        name='Rotación vs Stock'
                    )
                ])
                
                fig_rotation.update_layout(
                    title="Análisis de Rotación vs Stock",
                    xaxis_title="Stock Actual",
                    yaxis_title="Índice de Rotación",
                    template=self.theme
                )
                figures['rotation'] = fig_rotation

            # 3. Valor del inventario
            fig_value = go.Figure(data=[
                go.Bar(
                    x=inventory_data['categoria'],
                    y=inventory_data['valor_inventario'],
                    name='Valor del Inventario'
                )
            ])
            fig_value.update_layout(
                title="Valor del Inventario por Categoría",
                xaxis_title="Categoría",
                yaxis_title="Valor ($)",
                template=self.theme
            )
            figures['value'] = fig_value

            return figures

        except Exception as e:
            logger.error(f"Error al crear análisis de inventario: {str(e)}")
            return {'stock_dist': go.Figure(), 'rotation': go.Figure(), 'value': go.Figure()}


    def create_performance_dashboard(self, performance_data):
        """Crea dashboard de rendimiento"""
        try:
            if performance_data.empty:
                return go.Figure()

            # Creamos el layout con subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Ventas vs Tiempo",
                    "Distribución del Beneficio",
                    "Rendimiento por Categoría",
                    "Tendencia de Márgenes"
                ),
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )

            # 1. Ventas vs Tiempo - Agregamos por período
            ventas_tiempo = performance_data.groupby('periodo')['ingresos_totales'].sum().reset_index()
            fig.add_trace(
                go.Scatter(
                    x=ventas_tiempo['periodo'],
                    y=ventas_tiempo['ingresos_totales'],
                    name="Ingresos",
                    line=dict(color='blue')
                ),
                row=1, col=1
            )

            # 2. Distribución del Beneficio
            fig.add_trace(
                go.Box(
                    y=performance_data['beneficio_total'],
                    name="Beneficio"
                ),
                row=1, col=2
            )

            # 3. Rendimiento por Categoría - Agregamos por categoría
            cat_perf = performance_data.groupby('categoria').agg({
                'ingresos_totales': 'sum',
                'beneficio_total': 'sum'
            }).reset_index()
            
            fig.add_trace(
                go.Bar(
                    x=cat_perf['categoria'],
                    y=cat_perf['beneficio_total'],
                    name="Beneficio por Categoría"
                ),
                row=2, col=1
            )

            # 4. Tendencia de Márgenes - Agregamos por período y calculamos promedio
            margenes_tiempo = performance_data.groupby('periodo')['margen_porcentaje'].mean().reset_index()
            fig.add_trace(
                go.Scatter(
                    x=margenes_tiempo['periodo'],
                    y=margenes_tiempo['margen_porcentaje'],
                    name="Margen Promedio",
                    line=dict(color='green')
                ),
                row=2, col=2
            )

            # Actualizamos el layout
            fig.update_layout(
                height=800,
                title_text="Dashboard de Rendimiento",
                showlegend=True,
                template=self.theme
            )

            # Actualizamos los ejes
            fig.update_xaxes(title_text="Fecha", row=1, col=1)
            fig.update_xaxes(title_text="Categoría", row=2, col=1)
            fig.update_xaxes(title_text="Fecha", row=2, col=2)
            
            fig.update_yaxes(title_text="Ingresos ($)", row=1, col=1)
            fig.update_yaxes(title_text="Beneficio ($)", row=1, col=2)
            fig.update_yaxes(title_text="Beneficio Total ($)", row=2, col=1)
            fig.update_yaxes(title_text="Margen (%)", row=2, col=2)

            return fig

        except Exception as e:
            logger.error(f"Error al crear dashboard de rendimiento: {str(e)}")
            return go.Figure()
    
    # Añadir este método a la clase DashboardVisualizer en visualitations.py

    def create_forecast_visualization(self, historical_data: pd.DataFrame, forecast_data: pd.DataFrame) -> go.Figure:
        """
        Crea una visualización de las predicciones junto con los datos históricos.
        
        Args:
            historical_data (pd.DataFrame): DataFrame con los datos históricos
            forecast_data (pd.DataFrame): DataFrame con las predicciones
        
        Returns:
            go.Figure: Figura de Plotly con la visualización
        """
        try:
            fig = go.Figure()

            # Datos históricos
            fig.add_trace(
                go.Scatter(
                    x=historical_data['ds'],
                    y=historical_data['y'],
                    name="Ventas Históricas",
                    mode='lines+markers',
                    line=dict(color='#1f77b4', width=1),
                    marker=dict(size=4)
                )
            )

            # Predicción
            fig.add_trace(
                go.Scatter(
                    x=forecast_data['ds'],
                    y=forecast_data['yhat'],
                    name="Predicción",
                    mode='lines',
                    line=dict(color='#2ca02c', width=2, dash='dash')
                )
            )

            # Intervalo de confianza
            fig.add_trace(
                go.Scatter(
                    x=forecast_data['ds'].tolist() + forecast_data['ds'].tolist()[::-1],
                    y=forecast_data['yhat_upper'].tolist() + forecast_data['yhat_lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(0,176,246,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Intervalo de Confianza',
                    showlegend=True
                )
            )

            fig.update_layout(
                title="Predicción de Ventas",
                xaxis_title="Fecha",
                yaxis_title="Unidades Vendidas",
                hovermode='x unified',
                showlegend=True,
                template=self.theme
            )

            return fig

        except Exception as e:
            logger.error(f"Error al crear visualización de predicción: {e}")
            return go.Figure()  # Retorna una figura vacía en caso de error