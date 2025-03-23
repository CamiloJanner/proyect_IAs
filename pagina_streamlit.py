import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Cargar el modelo y el tokenizador
modelo = tf.keras.models.load_model("modelo_emocional.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

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

# Función para analizar el sentimiento
def analizar_sentimiento(frase):
    secuencia = tokenizer.texts_to_sequences([frase])
    padded = pad_sequences(secuencia, maxlen=10, padding='post')
    prediccion = modelo.predict(padded)
    return np.argmax(prediccion)  # Devuelve la clase con mayor probabilidad

# Interfaz en Streamlit
def main():
    st.markdown('<p style="font-family:Candara; color:White; font-size: 40px;">Análisis Emocional con DNN</p>', unsafe_allow_html=True)
    
    frase = st.text_area("Escribe un pensamiento o frase para analizar:")

    if st.button('Analizar Estado de Ánimo'):
        if frase.strip():
            resultado = analizar_sentimiento(frase)
            respuesta = random.choice(responses[resultado])  # Elegir respuesta aleatoria
            
            # Estilizar la respuesta según el sentimiento detectado
            color = "LightGreen" if resultado == 0 else "LightBlue" if resultado == 1 else "LightCoral"
            st.markdown(f'<p style="font-family:Candara; color:{color}; font-size: 20px;">{respuesta}</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="font-family:Candara; color:Red; font-size: 20px;">Por favor, ingresa una frase válida.</p>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
