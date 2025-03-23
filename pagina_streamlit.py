import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
import random
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# URL del modelo y el tokenizer en Google Drive
MODEL_URL = "https://drive.google.com/uc?id=147X4OsrSDUoyCCZCgdLP2KBnpSORUzm6"
TOKENIZER_URL = "https://drive.google.com/uc?id=1RkOdhGM7BUJWr0VLyj20VwlFjT0CCzN2"
MODEL_PATH = "sentiment140_model.h5"
TOKENIZER_PATH = "tokenizer_sentiment140.pkl"

# Función para descargar el modelo con `wget`
def descargar_archivo(file_id, output_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    if not os.path.exists(output_path):
        st.write(f"Descargando {output_path}...")
        os.system(f"wget --no-check-certificate '{url}' -O {output_path}")
    else:
        st.write(f"{output_path} ya existe.")

# Descargar modelo y tokenizer
descargar_archivo(MODEL_ID, MODEL_PATH)
descargar_archivo(TOKENIZER_ID, TOKENIZER_PATH)

# Verificar si los archivos se descargaron correctamente
if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
    st.write("✅ Descarga completada.")
else:
    st.write("❌ Error en la descarga.")

# Cargar modelo y tokenizador
modelo = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as handle:
    tokenizer = pickle.load(handle)

# Frases de respuesta según la predicción
responses = {
    0: [  # Positivo
        "Parece que estás de buen ánimo. Tu mensaje transmite optimismo y una actitud positiva.",
        "Tu mensaje refleja alegría y satisfacción. ¡Sigue disfrutando de ese buen estado de ánimo!",
        "Se percibe un tono entusiasta en tu mensaje. Parece que estás teniendo un buen día."
    ],
    1: [  # Neutro
        "Tu mensaje parece neutral, sin una carga emocional fuerte.",
        "No hay un sentimiento claro en tu mensaje. Puede ser algo informativo o una reflexión sin emoción específica.",
        "Parece que te mantienes en un estado equilibrado, sin expresar ni entusiasmo ni desánimo."
    ],
    2: [  # Negativo
        "Tu mensaje refleja algo de preocupación o malestar. Si necesitas desahogarte, estoy aquí para escuchar.",
        "Parece que algo no ha salido como esperabas. Recuerda que siempre hay nuevas oportunidades para mejorar las cosas.",
        "Detecto un tono de tristeza o frustración en tu mensaje. Espero que pronto te sientas mejor."
    ]
}

# Función de predicción
def predecir_sentimiento(texto):
    secuencia = tokenizer.texts_to_sequences([texto])
    secuencia_padded = pad_sequences(secuencia, maxlen=100)
    prediccion = modelo.predict(secuencia_padded)
    clase = np.argmax(prediccion)
    return clase, random.choice(responses[clase])

# UI en Streamlit
def main():
    st.title("Análisis de Sentimiento con IA")
    texto_usuario = st.text_area("Escribe un mensaje para analizar su sentimiento:")
    if st.button("Comprobar Sentimiento"):
        if texto_usuario.strip():
            sentimiento, respuesta = predecir_sentimiento(texto_usuario)
            st.write(f"**Resultado:** {['Positivo', 'Neutro', 'Negativo'][sentimiento]}")
            st.write(f"**Respuesta:** {respuesta}")
        else:
            st.write("Por favor, escribe un mensaje para analizar.")

if __name__ == "__main__":
    main()
