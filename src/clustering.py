import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import streamlit as st

class ProductClusterer:
    def __init__(self, database_url):
        self.engine = create_engine(database_url)
    
    def prepare_clustering_data(self):
        """
        Prepara datos para clustering con características relevantes
        """
        query = text("""
            SELECT 
                p.id,
                p.nombre, 
                p.categoria,
                p.precio_compra,
                p.precio_venta,
                p.stock_actual,
                COUNT(v.id) as total_ventas,
                COALESCE(SUM(v.cantidad), 0) as unidades_vendidas,
                COALESCE(SUM(v.precio_venta * v.cantidad), 0) as ingresos_totales
            FROM productos p
            LEFT JOIN ventas v ON p.id = v.producto_id
            GROUP BY p.id, p.nombre, p.categoria, p.precio_compra, p.precio_venta, p.stock_actual
        """)
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        return df
    
    def perform_clustering(self, n_clusters=4):
        """
        Realiza clustering de productos
        """
        # Preparar datos
        df = self.prepare_clustering_data()
        
        # Validar número máximo de clusters
        max_clusters = min(n_clusters, len(df) - 1)
        
        # Seleccionar características para clustering
        features = [
            'precio_compra', 
            'precio_venta', 
            'stock_actual', 
            'total_ventas', 
            'unidades_vendidas', 
            'ingresos_totales'
        ]
        
        # Manejar valores nulos y infinitos
        df[features] = df[features].fillna(0)
        df[features] = df[features].replace([np.inf, -np.inf], 0)
        
        # Escalar características
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features])
        
        # Aplicar clustering con K-Means
        kmeans = KMeans(
            n_clusters=max_clusters, 
            random_state=42, 
            n_init=10
        )
        df['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Reducir dimensionalidad para visualización
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        df['pca1'] = pca_result[:, 0]
        df['pca2'] = pca_result[:, 1]
        
        # Caracterizar clusters
        cluster_summary = df.groupby('cluster')[features + ['categoria']].agg({
            'precio_compra': 'mean',
            'precio_venta': 'mean', 
            'stock_actual': 'mean', 
            'total_ventas': 'mean', 
            'unidades_vendidas': 'mean', 
            'ingresos_totales': 'mean',
            'categoria': lambda x: x.mode().iloc[0]
        })
        
        return df, cluster_summary, kmeans.cluster_centers_
    
    def visualize_clusters(self, df):
        """
        Crea visualización de clusters
        """
        fig = px.scatter(
            df, 
            x='pca1', 
            y='pca2', 
            color='cluster', 
            hover_data=['nombre', 'categoria', 'precio_venta', 'unidades_vendidas'],
            title='Clustering de Productos',
            labels={'pca1': 'Dimensión 1', 'pca2': 'Dimensión 2'}
        )
        return fig
    
    def caracterizar_clusters(self, cluster_summary):
        """
        Genera descriptores de cada cluster
        """
        caracterizaciones = {}
        for cluster, datos in cluster_summary.iterrows():
            descripcion = f"""
            **Cluster {cluster}**:
            - Categoría Predominante: {datos['categoria']}
            - Precio de Compra Promedio: ${datos['precio_compra']:.2f}
            - Precio de Venta Promedio: ${datos['precio_venta']:.2f}
            - Stock Promedio: {datos['stock_actual']:.0f} unidades
            - Ventas Totales Promedio: {datos['total_ventas']:.0f}
            - Unidades Vendidas Promedio: {datos['unidades_vendidas']:.0f}
            - Ingresos Totales Promedio: ${datos['ingresos_totales']:.2f}
            """
            caracterizaciones[cluster] = descripcion
        return caracterizaciones

def pagina_clustering(db):
    st.title("Clasificación de Productos por Clustering")
    
    # Obtener número de productos
    clusterer = ProductClusterer(db.DATABASE_URL)
    df = clusterer.prepare_clustering_data()
    max_clusters = min(6, len(df) - 1)  # Límite máximo de 6 clusters
    
    # Selector de número de clusters
    n_clusters = st.slider(
        "Número de Clusters", 
        min_value=2, 
        max_value=max_clusters, 
        value=min(4, max_clusters), 
        step=1
    )
    
    # Realizar clustering
    try:
        df_clustered, cluster_summary, _ = clusterer.perform_clustering(n_clusters)
        
        # Visualización de clusters
        st.subheader("Visualización de Clusters")
        fig_clusters = clusterer.visualize_clusters(df_clustered)
        st.plotly_chart(fig_clusters)
        
        # Caracterización de clusters
        caracterizaciones = clusterer.caracterizar_clusters(cluster_summary)
        
        st.subheader("Descripción de Clusters")
        for cluster, descripcion in caracterizaciones.items():
            with st.expander(f"Cluster {cluster}"):
                st.markdown(descripcion)
        
        # Tabla de productos por cluster
        st.subheader("Productos en cada Cluster")
        cluster_table = df_clustered[['nombre', 'categoria', 'cluster', 'precio_venta', 'unidades_vendidas']]
        st.dataframe(cluster_table)
        
    except Exception as e:
        st.error(f"Error en clustering: {str(e)}")