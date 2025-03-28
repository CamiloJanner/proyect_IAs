import streamlit as st
import tensorflow as tf
import numpy as np
import os
import random
import pickle
import requests
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# IDs de los archivos en Google Drive
MODEL_ID = "1ooYFg0KB5zVT3YfLCrTz8KJ3QvjYjPgg"
TOKENIZER_ID = "1RkOdhGM7BUJWr0VLyj20VwlFjT0CCzN2"

# Rutas locales
MODEL_PATH = "modelo_sentimiento.h5"
TOKENIZER_PATH = "tokenizer.pickle"

# Descargar archivo solo si no existe
def descargar_archivo_drive(file_id, output_path):
    if not os.path.exists(output_path):
        URL = "https://drive.google.com/uc?export=download"
        with requests.Session() as session:
            response = session.get(URL, params={'id': file_id}, stream=True)
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    response = session.get(URL, params={'id': file_id, 'confirm': value}, stream=True)
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(32768):
                    f.write(chunk)

# Descargar y cargar modelo y tokenizador solo una vez
@st.cache_resource
def load_model():
    descargar_archivo_drive(MODEL_ID, MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

@st.cache_resource
def load_tokenizer():
    descargar_archivo_drive(TOKENIZER_ID, TOKENIZER_PATH)
    with open(TOKENIZER_PATH, "rb") as handle:
        return pickle.load(handle)

modelo = load_model()
tokenizer = load_tokenizer()

# Respuestas según la predicción
responses = {
    0: ["Parece que estás de buen ánimo. Tu mensaje transmite optimismo.", "Tu mensaje refleja alegría y satisfacción.", "Se percibe un tono entusiasta en tu mensaje."],
    1: ["Tu mensaje parece neutral, sin una carga emocional fuerte.", "No hay un sentimiento claro en tu mensaje.", "Parece que te mantienes en un estado equilibrado."],
    2: ["Tu mensaje refleja preocupación o malestar.", "Parece que algo no ha salido como esperabas.", "Detecto un tono de tristeza o frustración en tu mensaje."]
}

# Función de predicción optimizada
def predecir_sentimiento(texto):
    secuencia = tokenizer.texts_to_sequences([texto])
    secuencia_padded = pad_sequences(secuencia, maxlen=100)
    prediccion = modelo.predict(secuencia_padded)
    clase = np.argmax(prediccion)
    return clase, random.choice(responses[clase])

# UI en Streamlit sin recargar toda la página
st.title("Análisis de Sentimiento con IA")
texto_usuario = st.text_area("Escribe un mensaje para analizar su sentimiento:")

# Botón con `st.session_state` para evitar recargar la página
if "resultado" not in st.session_state:
    st.session_state.resultado = None
    st.session_state.respuesta = None

if st.button("Comprobar Sentimiento"):
    if texto_usuario.strip():
        sentimiento, respuesta = predecir_sentimiento(texto_usuario)
        st.session_state.resultado = ['Positivo', 'Neutro', 'Negativo'][sentimiento]
        st.session_state.respuesta = respuesta
    else:
        st.session_state.resultado = None
        st.session_state.respuesta = "Por favor, escribe un mensaje para analizar."

# Mostrar resultado sin refrescar la página
if st.session_state.resultado:
    st.write(f"**Resultado:** {st.session_state.resultado}")
    st.write(f"**Respuesta:** {st.session_state.respuesta}")
