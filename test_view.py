import streamlit as st
import pandas as pd
import os
import pickle
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    df.to_excel(writer, index=False, sheet_name="Sheet1")
    workbook = writer.book
    worksheet = writer.sheets["Sheet1"]
    format1 = workbook.add_format({"num_format": "0.00"})
    worksheet.set_column("A:A", None, format1)
    writer.close()
    processed_data = output.getvalue()
    return processed_data


def test_view():
    st.header("Teste")
    model_path = "models/my_model.pkl"
    if not os.path.isfile(model_path):
        st.write(
            "Nenhum modelo salvo no momento. Volte à tela de treinamento para criar um novo modelo."
        )
    else:
        uploaded_file = st.file_uploader("Escolha conjunto de teste")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df)
            test_button = st.button("Executar inferência")
            if test_button:
                model = pickle.load(open(model_path, "rb"))
                min_max_scaler = pickle.load(open("models/min_max.pkl", "rb"))
                X_test = df.drop("id", axis=1).copy()
                y_pred = model.predict(min_max_scaler.transform(X_test))
                X_test["pred"] = y_pred
                st.dataframe(X_test)
                df_xlsx = to_excel(X_test)
                st.download_button(
                    label="📥 Download Current Result",
                    data=df_xlsx,
                    file_name="df_test.xlsx",
                )
