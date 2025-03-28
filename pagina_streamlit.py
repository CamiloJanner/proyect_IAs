import streamlit as st
import tensorflow as tf
import numpy as np
import os
import random
import pickle
import requests
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from deep_translator import GoogleTranslator

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

# Respuestas aleatorias para cada sentimiento
responses = {
    0: [
        "¡Parece que estás de buen ánimo! Sigue disfrutando tu día. 😊",
        "Tu mensaje refleja una actitud positiva. ¡Sigue así! 🌟",
        "Se nota optimismo en tus palabras. ¡Eso es genial! 💪"
    ],
    1: [
        "Tu mensaje parece ser neutral, sin una emoción fuerte asociada. 🤔",
        "No detecto un sentimiento marcado en tu mensaje. ¿Tienes algo en mente? 🧐",
        "Parece que es un comentario equilibrado, sin inclinación emocional. 🎭"
    ],
    2: [
        "Percibo que podrías estar sintiéndote mal. Si necesitas hablar, aquí estoy. 🖤",
        "Tu mensaje suena algo negativo. Espero que todo mejore pronto. 🌧️",
        "Parece que no estás en tu mejor día. Recuerda que todo pasa. 💙"
    ]
}

# Función para predecir el sentimiento y dar una respuesta
def predecir_sentimiento(text):
    # Traducir el texto de español a inglés
    translated_text = GoogleTranslator(source='es', target='en').translate(text)
    
    sequence = tokenizer.texts_to_sequences([translated_text])
    padded = pad_sequences(sequence, maxlen=100, padding="post", truncating="post")
    prediccion = modelo.predict(padded)
    score = prediccion[0][0]
    
    if score < 0.4:
        clase = 2  # Negativo
    elif score > 0.6:
        clase = 0  # Positivo
    else:
        clase = 1  # Neutro
    
    return clase, random.choice(responses.get(clase, ["Error: Clase fuera de rango."]))

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
