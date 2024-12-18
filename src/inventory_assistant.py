import pandas as pd
import logging
from typing import Dict, List, Optional
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM

logger = logging.getLogger(__name__)

class InventoryAssistant:
    def __init__(self, db_manager):
        """Inicializa el asistente de inventario"""
        try:
            self.llm = OllamaLLM(model="llama3.2")
            self.db = db_manager
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Template para análisis general
            self.analysis_template = PromptTemplate(
                input_variables=["data"],
                template="""
                Como asistente experto en gestión de inventario de Amazon, analiza los siguientes datos:

                {data}

                Proporciona un análisis detallado incluyendo:
                1. Insights principales sobre el rendimiento actual
                2. Problemas potenciales que requieren atención
                3. Oportunidades de mejora
                4. Recomendaciones específicas y accionables
                
                Asegúrate de que el análisis sea específico y basado en los datos proporcionados.
                Quiero que el resultado sea siempre en idioma Español
                """
            )
            
            # Template para consultas específicas - Modificado para usar un solo input
            self.query_template = PromptTemplate(
                input_variables=["input"],
                template="""
                Responde la siguiente pregunta sobre el inventario y las ventas basándote en estos datos:
                
                Pregunta: {input}
                
                Proporciona una respuesta clara y concisa basada en los datos proporcionados.
                Quiero que el resultado sea siempre en idioma Español.
                """
            )
            
            # Template para recomendaciones
            self.recommendations_template = PromptTemplate(
                input_variables=["data"],
                template="""
                Basado en los siguientes datos de inventario y ventas, genera recomendaciones específicas:

                {data}

                Proporciona recomendaciones accionables para:
                1. Gestión de stock
                2. Optimización de precios
                3. Promociones sugeridas
                4. Acciones inmediatas necesarias

                Quiero que el resultado sea siempre en idioma Español
                """
            )
            
            self.analysis_chain = LLMChain(
                llm=self.llm,
                prompt=self.analysis_template,
                memory=self.memory,
                verbose=True
            )
            
            self.query_chain = LLMChain(
                llm=self.llm,
                prompt=self.query_template,
                memory=self.memory,
                verbose=True
            )
            
            self.recommendations_chain = LLMChain(
                llm=self.llm,
                prompt=self.recommendations_template,
                memory=self.memory,
                verbose=True
            )
            
        except Exception as e:
            logger.error(f"Error al inicializar el asistente: {e}")
            raise

    def _prepare_data_for_analysis(self, inventory_data, sales_data=None, metrics=None):
        """Prepara los datos para el análisis en un formato de texto legible"""
        data_str = f"Datos de Inventario:\n{inventory_data.to_string()}\n\n"
        
        if sales_data is not None:
            data_str += f"Datos de Ventas:\n{sales_data.to_string()}\n\n"
        
        if metrics is not None:
            data_str += "Métricas Principales:\n"
            for key, value in metrics.items():
                data_str += f"{key}: {value}\n"
        
        return data_str

    def get_inventory_insights(self) -> Dict:
        """Genera insights automáticos sobre el inventario actual"""
        try:
            # Obtener datos actualizados
            inventory_data = self.db.get_inventory_report()
            sales_data = self.db.get_sales_report()
            
            if inventory_data.empty or sales_data.empty:
                return {"error": "No hay suficientes datos para generar insights"}
            
            # Preparar métricas clave
            metrics = {
                "total_productos": len(inventory_data),
                "valor_total_inventario": inventory_data['valor_inventario'].sum(),
                "productos_sin_stock": len(inventory_data[inventory_data['stock_actual'] == 0]),
                "ventas_totales": sales_data['cantidad'].sum(),
                "ingresos_totales": (sales_data['cantidad'] * sales_data['precio_venta']).sum()
            }
            
            # Preparar datos para el modelo
            prepared_data = self._prepare_data_for_analysis(
                inventory_data, 
                sales_data, 
                metrics
            )
            
            # Generar análisis
            response = self.analysis_chain.run(data=prepared_data)
            
            return {
                "analysis": response,
                "metrics": metrics,
                "timestamp": pd.Timestamp.now()
            }
            
        except Exception as e:
            logger.error(f"Error al generar insights: {e}")
            return {"error": str(e)}

    def answer_query(self, question: str) -> str:
        try:
            # Obtener datos relevantes
            relevant_data = self._get_relevant_data(question)
            
            # Combinar la pregunta y los datos en un solo string
            combined_input = f"""
            Datos relevantes:
            {relevant_data}
            
            Pregunta del usuario:
            {question}
            """
            
            # Ejecutar la cadena con el input combinado
            response = self.query_chain.run(input=combined_input)
            
            return response
            
        except Exception as e:
            logger.error(f"Error al responder consulta: {e}")
            return f"Error al responder consulta: {e}"

        
    def get_recommendations(self) -> Dict:
        """Genera recomendaciones específicas de acciones a tomar"""
        try:
            # Obtener datos actualizados
            inventory_data = self.db.get_inventory_report()
            sales_data = self.db.get_sales_report()
            
            # Preparar datos para análisis
            prepared_data = self._prepare_data_for_analysis(
                inventory_data, 
                sales_data
            )
            
            response = self.recommendations_chain.run(data=prepared_data)
            
            return {
                "recommendations": response,
                "data_summary": {
                    "productos_stock_bajo": len(inventory_data[inventory_data['stock_actual'] <= 5]),
                    "productos_sin_ventas": len(inventory_data[inventory_data['ventas_totales'] == 0]),
                    "productos_alta_rotacion": len(inventory_data[inventory_data['ventas_totales'] > 100])
                },
                "timestamp": pd.Timestamp.now()
            }
            
        except Exception as e:
            logger.error(f"Error al generar recomendaciones: {e}")
            return {"error": str(e)}

    def _get_relevant_data(self, question: str) -> str:
        """Determina y obtiene los datos relevantes para una pregunta específica"""
        try:
            # Palabras clave para determinar qué datos necesitamos
            keywords = {
                "ventas": ["ventas", "vendido", "ingresos", "revenue"],
                "inventario": ["stock", "inventario", "existencias"],
                "productos": ["producto", "categoría", "SKU"],
                "precios": ["precio", "margen", "beneficio"]
            }
            
            needed_data = []
            question_lower = question.lower()
            
            # Identificar qué datos necesitamos
            for category, words in keywords.items():
                if any(word in question_lower for word in words):
                    needed_data.append(category)
            
            # Si no identificamos nada específico, traer datos básicos
            if not needed_data:
                needed_data = ["ventas", "inventario"]
            
            # Obtener los datos relevantes
            relevant_data = {}
            
            if "ventas" in needed_data:
                relevant_data["ventas"] = self.db.get_sales_report()
            if "inventario" in needed_data:
                relevant_data["inventario"] = self.db.get_inventory_report()
            
            # Convertir los datos a string de forma ordenada
            return "\n\n".join([
                f"{key.upper()}:\n{value.to_string()}"
                for key, value in relevant_data.items()
            ])
            
        except Exception as e:
            logger.error(f"Error al obtener datos relevantes: {e}")
            return "Error al obtener datos relevantes"