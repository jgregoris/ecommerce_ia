import streamlit as st
from datetime import datetime, timedelta

def pagina_asistente(db, assistant):
    st.title("Asistente Virtual de Inventario 🤖")
    
    # Tabs para diferentes funcionalidades
    tab1, tab2, tab3 = st.tabs([
        "💡 Insights Automáticos",
        "❓ Consultas",
        "📊 Recomendaciones"
    ])
    
    with tab1:
        st.subheader("Análisis Automático del Inventario")
        
        if st.button("Generar Insights", key="generate_insights"):
            with st.spinner("Analizando datos..."):
                insights = assistant.get_inventory_insights()
                
                if "error" in insights:
                    st.error(f"Error al generar insights: {insights['error']}")
                else:
                    # Mostrar métricas principales
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Productos en Catálogo",
                            f"{insights['metrics']['total_productos']:,}"
                        )
                    with col2:
                        st.metric(
                            "Valor del Inventario",
                            f"${insights['metrics']['valor_total_inventario']:,.2f}"
                        )
                    with col3:
                        st.metric(
                            "Productos Sin Stock",
                            f"{insights['metrics']['productos_sin_stock']:,}"
                        )
                    
                    # Mostrar análisis detallado
                    st.markdown("### Análisis Detallado")
                    st.write(insights['analysis'])
                    
                    # Timestamp del análisis
                    st.caption(f"Análisis generado el {insights['timestamp']}")
    
    with tab2:
        st.subheader("Consultas sobre el Inventario")
        
        # Ejemplos de preguntas
        st.markdown("""
        Ejemplos de preguntas que puedes hacer:
        - ¿Cuáles son los productos más vendidos este mes?
        - ¿Qué categorías tienen el stock más bajo?
        - ¿Cuál es el margen promedio por categoría?
        - ¿Qué productos necesitan reposición urgente?
        """)
        
        # Input para la pregunta
        question = st.text_input(
            "Haz tu pregunta sobre el inventario:",
            placeholder="Ejemplo: ¿Cuáles son los productos más vendidos?"
        )
        
        if question:
            if st.button("Obtener Respuesta", key="get_answer"):
                with st.spinner("Analizando tu pregunta..."):
                    response = assistant.answer_query(question)
                    if isinstance(response, str):
                        if response.startswith("Error"):
                            st.error(response)
                        else:
                            st.markdown("### Respuesta")
                            st.write(response)
                    else:
                        st.error("Respuesta inesperada del asistente")

    
    with tab3:
        st.subheader("Recomendaciones Específicas")
        
        if st.button("Generar Recomendaciones", key="generate_recommendations"):
            with st.spinner("Generando recomendaciones..."):
                recommendations = assistant.get_recommendations()
                
                if "error" in recommendations:
                    st.error(f"Error: {recommendations['error']}")
                else:
                    # Mostrar resumen
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Productos con Stock Bajo",
                            recommendations['data_summary']['productos_stock_bajo']
                        )
                    with col2:
                        st.metric(
                            "Productos sin Ventas",
                            recommendations['data_summary']['productos_sin_ventas']
                        )
                    with col3:
                        st.metric(
                            "Productos Alta Rotación",
                            recommendations['data_summary']['productos_alta_rotacion']
                        )
                    
                    # Mostrar recomendaciones
                    st.markdown("### Recomendaciones")
                    st.write(recommendations['recommendations'])
                    
                    st.caption(f"Recomendaciones generadas el {recommendations['timestamp']}")