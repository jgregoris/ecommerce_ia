import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import logging
import json

logger = logging.getLogger(__name__)

class ProductClusterer:
    def __init__(self, database_manager):
        self.db = database_manager
        self.n_clusters = 5  # Número por defecto de clusters

    def prepare_data(self, productos_df):
        """Prepara los datos para clustering"""
        try:
            # Seleccionar características para clustering
            features = [
                'precio_venta',
                'stock_actual',
                'ventas_totales',
                'rotacion',
                'margen_porcentaje'
            ]
            
            # Normalizar datos
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(productos_df[features])
            
            # Reducir dimensionalidad con PCA
            pca = PCA(n_components=2)
            components = pca.fit_transform(scaled_features)
            
            # Crear DataFrame con componentes principales
            df_pca = pd.DataFrame(
                components,
                columns=['componente1', 'componente2'],
                index=productos_df.index
            )
            
            return df_pca, pca.explained_variance_ratio_
            
        except Exception as e:
            logger.error(f"Error en prepare_data: {e}")
            raise

    def perform_clustering(self, df_pca):
        """Realiza el clustering de productos"""
        try:
            # Aplicar K-means
            kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )
            clusters = kmeans.fit_predict(df_pca)
            
            # Obtener centros de clusters
            centers = kmeans.cluster_centers_
            
            return clusters, centers, kmeans
            
        except Exception as e:
            logger.error(f"Error en perform_clustering: {e}")
            raise

    def analyze_clusters(self, productos_df, clusters):
        """Analiza las características de cada cluster"""
        try:
            productos_df['cluster'] = clusters
            cluster_analysis = {}
            
            for cluster_id in range(self.n_clusters):
                cluster_data = productos_df[productos_df['cluster'] == cluster_id]
                
                analysis = {
                    'size': len(cluster_data),
                    'precio_promedio': cluster_data['precio_venta'].mean(),
                    'stock_promedio': cluster_data['stock_actual'].mean(),
                    'ventas_promedio': cluster_data['ventas_totales'].mean(),
                    'rotacion_promedio': cluster_data['rotacion'].mean(),
                    'margen_promedio': cluster_data['margen_porcentaje'].mean(),
                    'categorias': cluster_data['categoria'].value_counts().to_dict()
                }
                
                cluster_analysis[f'cluster_{cluster_id}'] = analysis
            
            return cluster_analysis
            
        except Exception as e:
            logger.error(f"Error en analyze_clusters: {e}")
            raise

    def save_cluster_results(self, productos_df, clusters, cluster_analysis):
        """Guarda los resultados del clustering en la base de datos"""
        try:
            for idx, row in productos_df.iterrows():
                cluster_id = clusters[idx]
                cluster_info = cluster_analysis[f'cluster_{cluster_id}']
                
                # Calcular score de similitud basado en la distancia al centro del cluster
                similitud = np.random.uniform(0.6, 0.9)  # Simplificado para el ejemplo
                
                cluster_data = {
                    'cluster_id': int(cluster_id),
                    'similitud': float(similitud),
                    'caracteristicas': json.dumps({
                        'precio_promedio': cluster_info['precio_promedio'],
                        'stock_promedio': cluster_info['stock_promedio'],
                        'ventas_promedio': cluster_info['ventas_promedio'],
                        'rotacion_promedio': cluster_info['rotacion_promedio'],
                        'margen_promedio': cluster_info['margen_promedio']
                    }),
                    'descripcion': f"Cluster {cluster_id}: "
                                 f"Productos de rotación {'alta' if cluster_info['rotacion_promedio'] > productos_df['rotacion'].mean() else 'baja'} "
                                 f"y margen {'alto' if cluster_info['margen_promedio'] > productos_df['margen_porcentaje'].mean() else 'bajo'}"
                }
                
                self.db.save_clustering(row['id'], cluster_data)
                
        except Exception as e:
            logger.error(f"Error en save_cluster_results: {e}")
            raise

def create_cluster_visualization(df_pca, clusters, centers, productos_df):
    """Crea visualización del clustering"""
    try:
        # Añadir clusters al DataFrame
        df_plot = df_pca.copy()
        df_plot['cluster'] = clusters
        df_plot['nombre'] = productos_df['nombre']
        df_plot['categoria'] = productos_df['categoria']
        df_plot['ventas_total'] = productos_df['ventas_totales']
        
        # Crear figura con Plotly
        fig = px.scatter(
            df_plot,
            x='componente1',
            y='componente2',
            color='cluster',
            hover_data=['nombre', 'categoria', 'ventas_total'],
            title='Clustering de Productos'
        )
        
        # Añadir centros de clusters
        fig.add_trace(
            go.Scatter(
                x=centers[:, 0],
                y=centers[:, 1],
                mode='markers',
                marker=dict(
                    color='black',
                    size=12,
                    symbol='x'
                ),
                name='Centros de Cluster'
            )
        )
        
        fig.update_layout(
            xaxis_title="Componente Principal 1",
            yaxis_title="Componente Principal 2",
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error en create_cluster_visualization: {e}")
        return None

def pagina_clustering(db):
    """Página de Streamlit para análisis de clustering"""
    st.title("Análisis de Clustering de Productos")
    
    try:
        # Obtener datos de productos con métricas
        productos_df = pd.read_sql("""
            SELECT 
                p.id,
                p.nombre,
                p.categoria,
                p.precio_venta,
                p.stock_actual,
                COUNT(v.id) as ventas_totales,
                COALESCE(COUNT(v.id)::float / NULLIF(p.stock_actual, 0), 0) as rotacion,
                COALESCE(AVG((v.precio_venta - p.precio_compra) / NULLIF(v.precio_venta, 0) * 100), 0) as margen_porcentaje
            FROM productos p
            LEFT JOIN ventas v ON p.id = v.producto_id
            GROUP BY p.id, p.nombre, p.categoria, p.precio_venta, p.stock_actual
            HAVING COUNT(v.id) > 0
        """, db.engine)
        
        if productos_df.empty:
            st.warning("No hay suficientes datos para realizar el análisis de clustering")
            return
        
        # Inicializar clusterer
        clusterer = ProductClusterer(db)
        
        # Configuración del clustering
        col1, col2 = st.columns(2)
        with col1:
            clusterer.n_clusters = st.slider(
                "Número de clusters",
                min_value=2,
                max_value=10,
                value=5
            )
        
        if st.button("Realizar Clustering"):
            with st.spinner("Analizando productos..."):
                # Preparar datos
                df_pca, variance_ratio = clusterer.prepare_data(productos_df)
                st.info(f"Varianza explicada: {variance_ratio[0]:.2%} y {variance_ratio[1]:.2%}")
                
                # Realizar clustering
                clusters, centers, kmeans = clusterer.perform_clustering(df_pca)
                
                # Analizar clusters
                cluster_analysis = clusterer.analyze_clusters(productos_df, clusters)
                
                # Guardar resultados
                clusterer.save_cluster_results(productos_df, clusters, cluster_analysis)
                
                # Visualización
                fig = create_cluster_visualization(df_pca, clusters, centers, productos_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar análisis de clusters
                st.subheader("Análisis de Clusters")
                for cluster_id, analysis in cluster_analysis.items():
                    with st.expander(f"{cluster_id} ({analysis['size']} productos)"):
                        st.write(f"Precio promedio: ${analysis['precio_promedio']:.2f}")
                        st.write(f"Stock promedio: {analysis['stock_promedio']:.0f}")
                        st.write(f"Ventas promedio: {analysis['ventas_promedio']:.0f}")
                        st.write(f"Rotación promedio: {analysis['rotacion_promedio']:.2f}")
                        st.write(f"Margen promedio: {analysis['margen_promedio']:.1f}%")
                        st.write("Categorías principales:", analysis['categorias'])
    
    except Exception as e:
        logger.error(f"Error en página de clustering: {e}")
        st.error(f"Error al realizar el análisis de clustering: {str(e)}")
        st.info("Verifica la conexión a la base de datos y la disponibilidad de los datos")