import streamlit as st
import tensorflow as tf
import numpy as np
import os
import random
import pickle
import requests
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print(tf.__version__)

# IDs de los archivos en Google Drive
MODEL_ID = "147X4OsrSDUoyCCZCgdLP2KBnpSORUzm6"  # Reemplázalo con el ID correcto
TOKENIZER_ID = "1RkOdhGM7BUJWr0VLyj20VwlFjT0CCzN2"  # Reemplázalo con el ID correcto

# Rutas donde se guardarán los archivos descargados
MODEL_PATH = "modelo_sentimiento.h5"
TOKENIZER_PATH = "tokenizer.pickle"

def descargar_archivo_drive(file_id, output_path):
    URL = "https://drive.google.com/uc?export=download"

    with requests.Session() as session:
        response = session.get(URL, params={'id': file_id}, stream=True)
        
        # Google Drive bloquea descargas grandes con una advertencia
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                response = session.get(URL, params={'id': file_id, 'confirm': value}, stream=True)

        # Guardar el archivo descargado
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(32768):
                f.write(chunk)

    if os.path.exists(output_path):
        print(f"✅ {output_path} descargado correctamente.")
    else:
        print(f"❌ Error al descargar {output_path}. Verifica el ID del archivo.")


# Descargar modelo y tokenizer
descargar_archivo_drive(MODEL_ID, MODEL_PATH)
descargar_archivo_drive(TOKENIZER_ID, TOKENIZER_PATH)

# Verificar si los archivos están descargados
if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
    st.write("✅ Descarga completada.")
else:
    st.write("❌ Error en la descarga. Verifica los IDs.")

# Verificar si el modelo se descargó bien
if os.path.exists(MODEL_PATH):
    print(f"✅ Modelo descargado: {MODEL_PATH}")
    print(f"📏 Tamaño: {os.path.getsize(MODEL_PATH)} bytes")
else:
    print(f"❌ Error: El modelo {MODEL_PATH} no existe.")

# Cargar modelo y tokenizador
modelo = tf.keras.models.load_model(MODEL_PATH, compile=False)
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
