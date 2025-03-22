import streamlit as st
import pickle
import pandas as pd

# Cargar el modelo de análisis de sentimientos
with open("ModeloSentimiento.bin", 'rb') as f:
    modelo = pickle.load(f)

# Función para clasificar el sentimiento
def clasificar_sentimiento(sentimiento):
    if sentimiento == 1:
        st.markdown('<p style="font-family:Candara; color:Green; font-size: 20px;">Sentimiento Positivo</p>', unsafe_allow_html=True)
    elif sentimiento == 0:
        st.markdown('<p style="font-family:Candara; color:Gray; font-size: 20px;">Sentimiento Neutro</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="font-family:Candara; color:Red; font-size: 20px;">Sentimiento Negativo</p>', unsafe_allow_html=True)

# Función principal de la aplicación
def main():
    st.markdown('<p style="font-family:Candara; color:White; font-size: 40px;">Análisis de Sentimiento</p>', unsafe_allow_html=True)
    
    # Entrada del usuario
    frase = st.text_area("Escribe una frase para analizar el sentimiento:")
    
    if st.button('Comprobar Sentimiento'):
        if frase.strip():
            df = pd.DataFrame([frase], columns=['texto'])
            sentimiento = modelo.predict(df)[0]
            clasificar_sentimiento(sentimiento)
        else:
            st.markdown('<p style="font-family:Candara; color:Red; font-size: 20px;">Por favor, ingresa una frase válida.</p>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
