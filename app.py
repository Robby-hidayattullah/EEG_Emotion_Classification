import streamlit as st
import pandas as pd
from eeg_classifier import EEGClassifier

class EEGApp:
    def __init__(self):
        self.classifier = None
        self.df_test = None
        self.run_classification = False

    def display_title(self):
        st.markdown(
            "<h1 style='text-align: center; color: #4A6B8A; font-family: Arial, sans-serif;'>"
            "🧠 EmoBrain: EEG Emotion Classification 😃</h1>",
            unsafe_allow_html=True
        )

    def display_description(self):
        st.markdown(
            """
            <p style="text-align: center; font-size: 14px; color: #7A8A97; font-family: Arial, sans-serif;">
                EmoBrain is an application that uses machine learning to analyze and classify 
                a person's emotions based on EEG signals.
            </p>
            """,
            unsafe_allow_html=True
        )

    def upload_file(self):
        st.subheader("📂 Upload Your CSV File")
        uploaded_file = st.file_uploader("Upload file here", type=["csv"], label_visibility="collapsed")
        if uploaded_file:
            self.df_test = pd.read_csv(uploaded_file, sep=None, engine='python')
            self.df_test.reset_index(drop=True, inplace=True)

            st.subheader("📊 Data Preview")
            st.dataframe(self.df_test)

    def sidebar_configuration(self):
        st.sidebar.header("📌 Customize Your Classification")
        model_choice = st.sidebar.selectbox(
            "Choose Algorithm",
            ["Decision Tree", "K-Nearest Neighbors", "Support Vector Machine"],
            key="model_choice"
        )
        scaler_choice = st.sidebar.selectbox(
            "Choose Scaling Technique",
            ["Quantile Transformer", "MaxAbs Scaler", "Standard Scaler"],
            key="scaler_choice"
        )
        
        #buat objek baru dari kelas EEGClassifier dan menyimpannya dalam atribut self.classifier
        self.classifier = EEGClassifier(model_choice, scaler_choice)
        self.classifier.load_model()

        if self.df_test is not None:
            if st.sidebar.button("🚀 Run Classification", use_container_width=True):
                self.run_classification = True

    def run(self):
        self.display_title()
        self.display_description()
        self.upload_file()

        st.markdown(
            """
            <style>section[data-testid="stSidebar"] { background-color: #A0D8E6; }</style>
            """,
            unsafe_allow_html=True
        )
        self.sidebar_configuration()

        # Run, memanggil metode classify() dari objek EEGClassifier
        if self.run_classification:
            self.classifier.classify(self.df_test)

if __name__ == "__main__":
    app = EEGApp()
    app.run()
