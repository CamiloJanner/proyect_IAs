#importar librerias necesarias
import streamlit as st
import pickle
import pandas as pd

# Paso 1: Leer el archivo .bin
with open("Modelo_NB.bin", 'rb') as f:
        modelo = pickle.load(f)

#Función que mostrará el resultado
def clasificar (num):
    if num == 0:
        return "Normal"
    elif num == 1:
        return "Abnormal"

#Función que reproduce toda la página
def main ():
    #Título general
    st.title("Prueba de modelo")

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

    if st.button ('Predict'):
         st.success(clasificar(modelo.predict(df)))

if __name__ == '__main__':
    main()