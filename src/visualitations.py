import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class DashboardVisualizer:
    def __init__(self, theme="plotly_white"):
        self.theme = theme

    def create_main_dashboard(self, sales_data, inventory_data, predictions_data):
        """
        Crea el dashboard principal con múltiples gráficos
        """
        try:
            # Crear figura con subplots
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
            if not predictions_data.empty:
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

                # Predicción (eliminar la duplicación)
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
            print(f"Error al crear dashboard: {str(e)}")
            # Retornar una figura vacía en caso de error
            return go.Figure()

    def create_sales_analysis(self, sales_data):
        """
        Crea visualizaciones para el análisis de ventas
        """
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

            # 3. Estacionalidad
            sales_data['mes'] = pd.to_datetime(sales_data['fecha_venta']).dt.month
            ventas_mes = sales_data.groupby(['mes', 'categoria'])['cantidad'].mean().reset_index()
            
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
                title="Patrón Estacional de Ventas",
                xaxis_title="Mes",
                yaxis_title="Cantidad Promedio",
                template=self.theme
            )
            figures['seasonality'] = fig_season

            return figures

        except Exception as e:
            print(f"Error al crear análisis de ventas: {str(e)}")
            return {'trend': go.Figure(), 'distribution': go.Figure(), 'seasonality': go.Figure()}

    def create_inventory_analysis(self, inventory_data):
        """
        Crea visualizaciones para el análisis de inventario
        """
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
                template=self.theme
            )
            figures['stock_dist'] = fig_stock

            # 2. Rotación
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
                        )
                    )
                ])
                fig_rotation.update_layout(
                    title="Análisis de Rotación vs Stock",
                    xaxis_title="Stock Actual",
                    yaxis_title="Índice de Rotación",
                    template=self.theme
                )
                figures['rotation'] = fig_rotation

            return figures

        except Exception as e:
            print(f"Error al crear análisis de inventario: {str(e)}")
            return {'stock_dist': go.Figure(), 'rotation': go.Figure()}