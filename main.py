import streamlit as st
from train_view import train_view
from test_view import test_view

st.title("Modelo de previsão de categoria de preço de Smartphones")

view_selection = st.radio("Escolha a opção de tela", ["Treinamento", "Teste"])
if view_selection == "Treinamento":
    train_view()
else:
    test_view()
