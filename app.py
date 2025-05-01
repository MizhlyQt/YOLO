import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys
from PIL import Image

# Configuración de página (DEBE SER EL PRIMER COMANDO STREAMLIT)
st.set_page_config(
    page_title="Detección de Objetos",
    page_icon="🔍",
    layout="wide"
)

# Título y selector de método
st.title("🔍 Detección de Objetos YOLOv5")
metodo = st.radio(
    "Seleccione el método de entrada:",
    ["📷 Usar cámara", "🖼️ Subir imagen"],
    horizontal=True,
    index=0
)

# Función para cargar el modelo (compatible con torch 1.12.0)
@st.cache_resource
def load_model():
    try:
        # Forma compatible con tus versiones específicas
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', 
                             pretrained=True, 
                             trust_repo=True)
        return model
    except Exception as e:
        st.error(f"Error al cargar modelo: {str(e)}")
        return None

# Cargar modelo
model = load_model()

if model:
    # Sidebar con TODOS los parámetros originales
    with st.sidebar:
        st.title("⚙️ Parámetros")
        
        # Configuración de detección
        st.subheader("Configuración básica")
        model.conf = st.slider("Umbral confianza", 0.0, 1.0, 0.25, 0.01)
        model.iou = st.slider("Umbral IoU", 0.0, 1.0, 0.45, 0.01)
        
        # Opciones avanzadas
        st.subheader("Opciones avanzadas")
        model.max_det = st.number_input("Máx. detecciones", 10, 2000, 1000)
        model.agnostic = st.checkbox("NMS agnóstico", False)
        model.multi_label = st.checkbox("Múltiples etiquetas", False)
        
        # Filtro para cámara (solo visible cuando corresponda)
        if metodo == "📷 Usar cámara":
            st.subheader("Procesamiento cámara")
            filtro_camara = st.checkbox("Aplicar filtro inverso")

    # Obtener imagen según método seleccionado
    img = None
    if metodo == "📷 Usar cámara":
        img_file = st.camera_input("Tome una foto")
        if img_file:
            bytes_data = img_file.getvalue()
            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            if filtro_camara:
                img = cv2.bitwise_not(img)
    else:
        img_file = st.file_uploader("Suba una imagen", type=["jpg", "png", "jpeg"])
        if img_file:
            bytes_data = img_file.read()
            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # Procesamiento de detección
    if img is not None:
        with st.spinner("Analizando imagen..."):
            try:
                # Convertir y realizar detección
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = model(img_rgb)
                
                # Mostrar resultados en 2 columnas
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Detecciones visuales")
                    results.render()  # Añade las cajas a la imagen
                    st.image(img_rgb, use_column_width=True)
                
                with col2:
                    st.subheader("Resultados numéricos")
                    # DataFrame con resultados
                    df = results.pandas().xyxy[0]
                    df = df[['name', 'confidence']].rename(columns={
                        'name': 'Objeto',
                        'confidence': 'Confianza'
                    })
                    
                    st.dataframe(
                        df.sort_values('Confianza', ascending=False),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Resumen estadístico
                    st.subheader("Resumen")
                    summary = df['Objeto'].value_counts().reset_index()
                    st.bar_chart(summary.set_index('Objeto'))
                    
                    # Botón de descarga
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Descargar resultados",
                        data=csv,
                        file_name='resultados.csv',
                        mime='text/csv'
                    )
            
            except Exception as e:
                st.error(f"Error en detección: {str(e)}")

# Mensaje si no hay modelo
else:
    st.error("""
    No se pudo cargar el modelo. Verifique:
    1. Requerimientos instalados (torch==1.12.0)
    2. Conexión a internet para descargar pesos
    """)

# Pie de página
st.markdown("---")
st.caption("Aplicación de detección de objetos | YOLOv5 | torch==1.12.0")
