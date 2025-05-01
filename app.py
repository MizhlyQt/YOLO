import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# Configuración de página Streamlit
st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide"
)

# Función para cargar el modelo 
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

# Título y descripción 
st.title("🔍 Detección de Objetos en Imágenes")
st.markdown("""
Esta aplicación utiliza YOLOv5 para detectar objetos. Seleccione el método de entrada:
""")

# Selector de método 
option = st.radio(
    "Fuente de imagen:",
    ["📷 Usar cámara", "🖼️ Subir imagen"],
    horizontal=True
)

# Cargar el modelo 
with st.spinner("Cargando modelo YOLOv5..."):
    model = load_yolov5_model()

if model:
    # Sidebar 
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

    # Contenedor principal 
    main_container = st.container()
    
    with main_container:
        # Obtención de imagen según selección 
        if option == "📷 Usar cámara":
            img_file_buffer = st.camera_input("Toma una foto")
        else:
            img_file_buffer = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

        if img_file_buffer is not None:
            # Procesamiento de imagen 
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Detección 
            with st.spinner("Detectando objetos..."):
                try:
                    results = model(cv2_img)
                except Exception as e:
                    st.error(f"Error durante la detección: {str(e)}")
                    st.stop()
            
            # Visualización 
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
                        st.bar_chart(
                            df.set_index('Categoría')['Cantidad'],
                            color='#83eb2c'  
                        )
                    else:
                        st.info("No se detectaron objetos con los parámetros actuales.")
            except Exception as e:
                st.error(f"Error al procesar los resultados: {str(e)}")

# Mensaje error 
else:
    st.error("No se pudo cargar el modelo. Por favor verifica las dependencias e inténtalo nuevamente.")
    st.stop()

# Pie de página 
st.markdown("---")
st.caption("""
**Acerca de la aplicación**: Esta aplicación utiliza YOLOv5 para detección de objetos en tiempo real.
Desarrollada con Streamlit y PyTorch.
""")
