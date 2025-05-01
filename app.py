import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# Configuración de página Streamlit (DEBE SER LA PRIMERA LÍNEA DE STREAMLIT)
st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide"
)

# Función para cargar el modelo YOLOv5
@st.cache_resource
def load_yolov5_model():
    try:
        # Cargar el modelo con torch hub (compatible con torch 1.12.0)
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("""
        Recomendaciones:
        1. Verifica que tienes conexión a internet
        2. Asegúrate de tener instaladas las dependencias
        """)
        return None

# Título y descripción
st.title("🔍 Detección de Objetos en Imágenes")
st.markdown("""
Seleccione el método de entrada y ajuste los parámetros en la barra lateral.
""")

# Opción para elegir entre cámara o subir imagen
input_method = st.radio(
    "Seleccione el método de entrada:",
    ["📷 Usar cámara", "🖼️ Subir imagen"],
    horizontal=True
)

# Cargar el modelo
model = load_yolov5_model()

if model:
    # Sidebar para parámetros (MANTENIENDO TODAS LAS OPCIONES ORIGINALES)
    st.sidebar.title("Parámetros")
    
    with st.sidebar:
        st.subheader('Configuración de detección')
        model.conf = st.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01)
        model.iou = st.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
        st.caption(f"Confianza: {model.conf:.2f} | IoU: {model.iou:.2f}")
        
        st.subheader('Opciones avanzadas')
        try:
            model.agnostic = st.checkbox('NMS class-agnostic', False)
            model.multi_label = st.checkbox('Múltiples etiquetas por caja', False)
            model.max_det = st.number_input('Detecciones máximas', 10, 2000, 1000, 10)
        except:
            st.warning("Algunas opciones avanzadas no están disponibles")

    # Procesamiento según el método seleccionado
    if input_method == "📷 Usar cámara":
        img_file_buffer = st.camera_input("Toma una foto")
        if img_file_buffer is not None:
            # Procesamiento para cámara (MANTENIENDO EL FILTRO ORIGINAL)
            with st.sidebar:
                st.subheader("Procesamiento para Cámara")
                filtro = st.radio("Filtro para imagen con cámara", ('Sí', 'No'))
            
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            if filtro == 'Sí':
                cv2_img = cv2.bitwise_not(cv2_img)
    else:
        # Opción para subir imagen
        img_file_buffer = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])
        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    if img_file_buffer is not None:
        # Realizar detección (MANTENIENDO EL PROCESAMIENTO ORIGINAL)
        with st.spinner("Detectando objetos..."):
            try:
                results = model(cv2_img)
            except Exception as e:
                st.error(f"Error durante la detección: {str(e)}")
                st.stop()
        
        # Mostrar resultados (MANTENIENDO LA VISUALIZACIÓN ORIGINAL)
        try:
            predictions = results.pred[0]
            boxes = predictions[:, :4]
            scores = predictions[:, 4]
            categories = predictions[:, 5]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Imagen con detecciones")
                results.render()
                st.image(cv2_img, channels='BGR', use_column_width=True)
            
            with col2:
                st.subheader("Objetos detectados")
                
                label_names = model.names
                category_count = {}
                for category in categories:
                    category_idx = int(category.item()) if hasattr(category, 'item') else int(category)
                    category_count[category_idx] = category_count.get(category_idx, 0) + 1
                
                data = []
                for category, count in category_count.items():
                    label = label_names[category]
                    confidence = scores[categories == category].mean().item() if len(scores) > 0 else 0
                    data.append({
                        "Categoría": label,
                        "Cantidad": count,
                        "Confianza promedio": f"{confidence:.2f}"
                    })
                
                if data:
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True)
                    st.bar_chart(df.set_index('Categoría')['Cantidad'])
                else:
                    st.info("No se detectaron objetos")
        except Exception as e:
            st.error(f"Error al procesar resultados: {str(e)}")

    # Pie de página original
    st.markdown("---")
    st.caption("""
    **Acerca de la aplicación**: Esta aplicación utiliza YOLOv5 para detección de objetos.
    Desarrollada con Streamlit y PyTorch.
    """)
else:
    st.error("No se pudo cargar el modelo. Verifica las dependencias.")
