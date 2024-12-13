import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class DashboardVisualizer:
    def __init__(self, theme="plotly_white"):
        self.theme = theme
        
    def create_main_dashboard(self, 
                            sales_data: pd.DataFrame, 
                            inventory_data: pd.DataFrame, 
                            predictions_data: pd.DataFrame) -> go.Figure:
        """
        Crea el dashboard principal con múltiples gráficos
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Tendencia de Ventas y Predicción",
                "Ventas por Categoría",
                "Análisis de Stock",
                "Rotación de Inventario",
                "Top Productos",
                "Rentabilidad por Categoría"
            ),
            specs=[
                [{"colspan": 2}, None],
                [{}, {}],
                [{}, {}]
            ],
            vertical_spacing=0.12
        )

        # 1. Tendencia de Ventas y Predicción
        fig.add_trace(
            go.Scatter(
                x=sales_data['fecha_venta'],
                y=sales_data['cantidad'],
                name="Ventas Reales",
                line=dict(color="#8884d8", width=2)
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=predictions_data['ds'],
                y=predictions_data['yhat'],
                name="Predicción",
                line=dict(color="#82ca9d", width=2, dash='dash')
            ),
            row=1, col=1
        )

        # 2. Ventas por Categoría
        ventas_categoria = sales_data.groupby('categoria')['cantidad'].sum().reset_index()
        fig.add_trace(
            go.Bar(
                x=ventas_categoria['categoria'],
                y=ventas_categoria['cantidad'],
                name="Ventas",
                marker_color="#8884d8"
            ),
            row=2, col=1
        )

        # 3. Análisis de Stock
        fig.add_trace(
            go.Bar(
                x=inventory_data['categoria'],
                y=inventory_data['stock_actual'],
                name="Stock Actual",
                marker_color="#82ca9d"
            ),
            row=2, col=2
        )

        # 4. Rotación de Inventario
        rotacion = inventory_data.copy()
        rotacion['rotacion'] = (inventory_data['ventas_totales'] / 
                               inventory_data['stock_actual']).fillna(0)
        fig.add_trace(
            go.Bar(
                x=rotacion['categoria'],
                y=rotacion['rotacion'],
                name="Rotación",
                marker_color="#ffc658"
            ),
            row=3, col=1
        )

        # 5. Rentabilidad por Categoría
        rentabilidad = sales_data.groupby('categoria').agg({
            'beneficio': 'sum'
        }).reset_index()
        fig.add_trace(
            go.Bar(
                x=rentabilidad['categoria'],
                y=rentabilidad['beneficio'],
                name="Rentabilidad",
                marker_color="#ff7c43"
            ),
            row=3, col=2
        )

        # Actualizar layout
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text="Dashboard de Ventas e Inventario",
            template=self.theme
        )

        return fig

    def create_sales_analysis(self, sales_data: pd.DataFrame) -> Dict[str, go.Figure]:
        """
        Crea visualizaciones específicas para el análisis de ventas
        """
        figures = {}

        # 1. Tendencia temporal de ventas
        figures['trend'] = px.line(
            sales_data,
            x='fecha_venta',
            y='cantidad',
            color='categoria',
            title="Tendencia de Ventas por Categoría",
            template=self.theme
        )

        # 2. Distribución de ventas por producto
        ventas_producto = sales_data.groupby('producto').agg({
            'cantidad': 'sum',
            'beneficio': 'sum'
        }).reset_index()

        figures['distribution'] = px.scatter(
            ventas_producto,
            x='cantidad',
            y='beneficio',
            size='cantidad',
            color='cantidad',
            hover_data=['producto'],
            title="Distribución de Ventas y Beneficios por Producto",
            template=self.theme
        )

        # 3. Análisis de estacionalidad
        sales_data['mes'] = pd.to_datetime(sales_data['fecha_venta']).dt.month
        ventas_mes = sales_data.groupby('mes')['cantidad'].mean().reset_index()

        figures['seasonality'] = px.line(
            ventas_mes,
            x='mes',
            y='cantidad',
            title="Patrón Estacional de Ventas",
            template=self.theme
        )

        return figures

    def create_inventory_analysis(self, inventory_data: pd.DataFrame) -> Dict[str, go.Figure]:
        """
        Crea visualizaciones para el análisis de inventario
        """
        figures = {}

        # 1. Distribución de stock
        figures['stock_dist'] = px.bar(
            inventory_data,
            x='categoria',
            y='stock_actual',
            color='stock_actual',
            title="Distribución de Stock por Categoría",
            template=self.theme
        )

        # 2. Análisis de rotación
        inventory_data['rotacion'] = (inventory_data['ventas_totales'] / 
                                    inventory_data['stock_actual']).fillna(0)

        figures['rotation'] = px.scatter(
            inventory_data,
            x='stock_actual',
            y='rotacion',
            color='categoria',
            size='valor_inventario',
            hover_data=['nombre'],
            title="Análisis de Rotación vs Stock",
            template=self.theme
        )

        return figures

    def create_anomaly_visualization(self, 
                                   data: pd.DataFrame, 
                                   anomalies: pd.DataFrame) -> go.Figure:
        """
        Crea visualización para la detección de anomalías
        """
        fig = go.Figure()

        # Datos principales
        fig.add_trace(
            go.Scatter(
                x=data['fecha'],
                y=data['valor'],
                name="Valores Normales",
                mode='lines+markers',
                line=dict(color="#8884d8")
            )
        )

        # Anomalías
        fig.add_trace(
            go.Scatter(
                x=anomalies['fecha'],
                y=anomalies['valor'],
                name="Anomalías",
                mode='markers',
                marker=dict(
                    color='red',
                    size=10,
                    symbol='x'
                )
            )
        )

        fig.update_layout(
            title="Detección de Anomalías en Ventas",
            xaxis_title="Fecha",
            yaxis_title="Valor",
            template=self.theme
        )

        return fig

    def create_clustering_visualization(self, 
                                     df_clusters: pd.DataFrame, 
                                     cluster_centers: np.ndarray) -> go.Figure:
        """
        Crea visualización para el análisis de clustering
        """
        fig = px.scatter(
            df_clusters,
            x='componente1',
            y='componente2',
            color='cluster',
            hover_data=['nombre', 'categoria', 'ventas_total'],
            title="Clustering de Productos",
            template=self.theme
        )

        # Añadir centros de clusters
        fig.add_trace(
            go.Scatter(
                x=cluster_centers[:, 0],
                y=cluster_centers[:, 1],
                mode='markers',
                marker=dict(
                    color='red',
                    size=15,
                    symbol='star'
                ),
                name='Centros de Cluster'
            )
        )

        fig.update_layout(
            xaxis_title="Componente Principal 1",
            yaxis_title="Componente Principal 2"
        )

        return fig