import streamlit as st
import tensorflow as tf
import gdown
import numpy as np
import random
import pickle

# URL de descarga del modelo y tokenizer (reemplaza con tus propios IDs de Google Drive)
model_file_id = "147X4OsrSDUoyCCZCgdLP2KBnpSORUzm6"
tokenizer_file_id = "1RkOdhGM7BUJWr0VLyj20VwlFjT0CCzN2"

model_url = f"https://drive.google.com/uc?id={model_file_id}"
tokenizer_url = f"https://drive.google.com/uc?id={tokenizer_file_id}"

model_output = "sentiment140_model.h5"
tokenizer_output = "tokenizer_sentiment140.pk"

gdown.download(model_url, model_output, use_cookies=False)
gdown.download(tokenizer_url, tokenizer_output, use_cookies=False)

# Cargar el modelo y el tokenizer
modelo = tf.keras.models.load_model(model_output)
with open(tokenizer_output, 'rb') as handle:
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

def analizar_sentimiento(texto):
    """Preprocesa el texto usando el tokenizer y predice el sentimiento."""
    secuencia = tokenizer.texts_to_sequences([texto])
    entrada = tf.keras.preprocessing.sequence.pad_sequences(secuencia, maxlen=100)
    prediccion = np.argmax(modelo.predict(entrada))
    return prediccion

def main():
    st.title("Análisis de Sentimiento con IA")
    
    # Cuadro de entrada
    texto_usuario = st.text_area("Escribe una frase y presiona el botón para analizar su sentimiento:")
    
    if st.button("Comprobar Sentimiento"):
        if texto_usuario.strip():
            sentimiento = analizar_sentimiento(texto_usuario)
            respuesta = random.choice(responses[sentimiento])
            
            # Mostrar resultado
            if sentimiento == 0:
                st.markdown(f'<p style="color:green; font-size:20px;">{respuesta}</p>', unsafe_allow_html=True)
            elif sentimiento == 1:
                st.markdown(f'<p style="color:gray; font-size:20px;">{respuesta}</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p style="color:red; font-size:20px;">{respuesta}</p>', unsafe_allow_html=True)
        else:
            st.warning("Por favor, ingresa un texto antes de comprobar el sentimiento.")

if __name__ == '__main__':
    main()
