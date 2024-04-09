#importar librerias necesarias
import streamlit as st
import pickle
import pandas as pd

# Paso 1: Leer el archivo .bin
with open("ModeloRegresion.bin", 'rb') as f:
        modelo = pickle.load(f)

#Función que mostrará el resultado
def clasificar (num):
    if num == 0:
        original_title = '<p style="font-family:Candara; color:Green; font-size: 20px;">El cliente no va a desertar</p>'
        st.markdown(original_title, unsafe_allow_html=True)
    elif num == 1:
        original_title = '<p style="font-family:Candara; color:Red; font-size: 20px;">El cliente va a desertar</p>'
        st.markdown(original_title, unsafe_allow_html=True)

#Función que reproduce toda la página
def main ():
    #Título general
    original_title = '<p style="font-family:Candara; color:White; font-size: 40px;">Prueba de modelo</p>'
    st.markdown(original_title, unsafe_allow_html=True)

    #Titulo del sidebar
    st.sidebar.header("Insertar los datos")

    #Función donde el usuario selecciona los datos para que el modelo haga la predicción
    def user_input_parameters():
        
        antig = st.sidebar.text_input("ANTIG")
        comp = st.sidebar.slider("COMP", 4414, 18338)
        prom = st.sidebar.text_input("PROM")
        categ = st.sidebar.text_input("CATEG")
        comint = st.sidebar.slider("COMINT", 1576, 57221)
        compres = st.sidebar.slider("COMPPRES",17517 , 91669)
        rate = st.sidebar.text_input("RATE")
        visit = st.sidebar.slider("VISIT", 2, 130)
        diasinq = st.sidebar.slider("DIASSINQ", 299, 1785)
        tasaret = st.sidebar.text_input("TASARET")
        numq = st.sidebar.text_input("NUMQ")
        retre = st.sidebar.text_input("RETRE")

        data = {  'ANTIG': antig,
                'COMP': comp,
                'PROM': prom,
                'CATEG': categ,
                'COMINT': comint,
                'COMPPRES': compres,
                'RATE': rate,
                'VISIT': visit,
                'DIASSINQ': diasinq,
                'TASARET': tasaret,
                'NUMQ': numq,
                'RETRE': retre}

        features = pd.DataFrame(data, index = [0])
        return features
        

    df = user_input_parameters()

    st.subheader("Modelo Naive Bayes")
    st.subheader("Datos insertados por el usuario")
    st.write(df)

    try:
        if st.button ('Predict'):
            clasificar(modelo.predict(df))
    except ValueError:
            original_title = '<p style="font-family:Candara; color:Red; font-size: 20px;">No ha insertado los datos de acuerdo al modelo</p>'
            st.markdown(original_title, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
