import streamlit as st
import pandas as pd
import numpy as np
from train import train_model
import plotly.graph_objs as go
import plotly.express as px


def train_view():
    st.header("Treinamento")
    uploaded_file = st.file_uploader("Escolha conjunto de treinamento")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.DataFrame()

    if uploaded_file:
        with st.expander("Dados"):
            st.dataframe(df)
        with st.expander("Proporção das classes"):
            st.plotly_chart(
                px.bar(
                    df["price_range"].value_counts().reset_index(),
                    x="price_range",
                    y="count",
                    title="Contagem de classes",
                )
            )

        train_button = st.button("Treinar modelo")
        if train_button:
            cm, accuracy, model = train_model(df)
            st.write(f"Acurácia no conjunto de testes = {accuracy: 0.2f}")
            layout = go.Layout(
                title="Confusion Matrix",
                xaxis=dict(title="Predicted Label"),
                yaxis=dict(title="True Label"),
            )
            classes = [f"{value}" for value in model.classes_]
            trace = go.Heatmap(z=cm, x=classes, y=classes, colorscale="Viridis")
            fig = go.Figure(data=[trace], layout=layout)
            st.plotly_chart(fig)
