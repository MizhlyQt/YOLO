import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# Configuración de página Streamlit (PRIMERA LÍNEA SIEMPRE)
st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide"
)

# Función para cargar el modelo YOLOv5 (EXACTAMENTE IGUAL)
@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        import yolov5
        
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            try:
                model = yolov5.load(model_path)
                return model
            except Exception as e:
                st.warning(f"Intentando método alternativo de carga...")
                
                current_dir = os.path.dirname(os.path.abspath(__file__))
                if current_dir not in sys.path:
                    sys.path.append(current_dir)
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                return model
    
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("""
        1. Instalar versión compatible:
           pip install torch==1.12.0 torchvision==0.13.0
           pip install yolov5==7.0.9
        2. Asegúrate de tener el archivo del modelo
        3. Intenta descargar el modelo directamente
        """)
        return None

# Título y descripción (IGUAL AL ORIGINAL)
st.title("🔍 Detección de Objetos en Imágenes")
st.markdown("""
Esta aplicación utiliza YOLOv5 para detectar objetos en imágenes capturadas con tu cámara.
Ajusta los parámetros en la barra lateral para personalizar la detección.
""")

# --- NUEVA FUNCIONALIDAD AÑADIDA --- #
option = st.radio(
    "Seleccione el método de entrada:",
    ["📷 Usar cámara", "🖼️ Subir imagen"],
    horizontal=True
)
# ----------------------------------- #

# Cargar el modelo (IGUAL AL ORIGINAL)
with st.spinner("Cargando modelo YOLOv5..."):
    model = load_yolov5_model()

if model:
    # Sidebar (CONSERVANDO TODOS LOS PARÁMETROS ORIGINALES)
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

    # Contenedor principal (CONSERVANDO TODA LA LÓGICA ORIGINAL)
    main_container = st.container()
    
    with main_container:
        # --- MODIFICACIÓN PARA SOPORTAR AMBAS OPCIONES --- #
        if option == "📷 Usar cámara":
            picture = st.camera_input("Capturar imagen", key="camera")
        else:
            picture = st.file_uploader("Subir imagen", type=["png", "jpg", "jpeg"])
        # ------------------------------------------------- #
        
        if picture:
            # Procesamiento (CONSERVANDO TODO EL CÓDIGO ORIGINAL)
            bytes_data = picture.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # MANTENIENDO EL FILTRO ORIGINAL SOLO PARA CÁMARA
            if option == "📷 Usar cámara":
                with st.sidebar:
                    st.subheader("Procesamiento para Cámara")
                    filtro = st.radio("Filtro para imagen con cámara", ('Sí', 'No'))
                
                if filtro == 'Sí':
                    cv2_img = cv2.bitwise_not(cv2_img)
            
            # DETECCIÓN ORIGINAL (SIN CAMBIOS)
            with st.spinner("Detectando objetos..."):
                try:
                    results = model(cv2_img)
                except Exception as e:
                    st.error(f"Error durante la detección: {str(e)}")
                    st.stop()
            
            # VISUALIZACIÓN ORIGINAL (SIN CAMBIOS)
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
                        if category_idx in category_count:
                            category_count[category_idx] += 1
                        else:
                            category_count[category_idx] = 1
                    
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
                        st.info("No se detectaron objetos con los parámetros actuales.")
            except Exception as e:
                st.error(f"Error al procesar los resultados: {str(e)}")

# Mensaje original si no hay modelo (SIN CAMBIOS)
else:
    st.error("No se pudo cargar el modelo. Por favor verifica las dependencias e inténtalo nuevamente.")
    st.stop()

# Pie de página original (SIN CAMBIOS)
st.markdown("---")
st.caption("""
**Acerca de la aplicación**: Esta aplicación utiliza YOLOv5 para detección de objetos en tiempo real.
Desarrollada con Streamlit y PyTorch.
""")
