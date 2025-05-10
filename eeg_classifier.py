import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import QuantileTransformer, MaxAbsScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

LABEL_MAPPING = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}

class EEGClassifier:
    def __init__(self, model_choice, scaler_choice):
        self.model_choice = model_choice
        self.scaler_choice = scaler_choice
        self.model = None
        self.scaler = None

    def load_model(self):
        model_file_mapping = {
            ("K-Nearest Neighbors", "Quantile Transformer"): "QuantileTransformer_knn_best.pkl",
            ("K-Nearest Neighbors", "MaxAbs Scaler"): "MaxAbsScaler_knn_best.pkl",
            ("K-Nearest Neighbors", "Standard Scaler"): "StandardScaler_knn_best.pkl",
            ("Support Vector Machine", "Quantile Transformer"): "QuantileTransformer_svc_best.pkl",
            ("Support Vector Machine", "MaxAbs Scaler"): "MaxAbsScaler_svc_best.pkl",
            ("Support Vector Machine", "Standard Scaler"): "StandardScaler_svc_best.pkl",
            ("Decision Tree", "Quantile Transformer"): "QuantileTransformer_dt_best.pkl",
            ("Decision Tree", "MaxAbs Scaler"): "MaxAbsScaler_dt_best.pkl",
            ("Decision Tree", "Standard Scaler"): "StandardScaler_dt_best.pkl",
        }

        scaler_mapping = {
            "Quantile Transformer": QuantileTransformer,
            "MaxAbs Scaler": MaxAbsScaler,
            "Standard Scaler": StandardScaler
        }

        key = (self.model_choice, self.scaler_choice)
        if key in model_file_mapping:
            model_file = model_file_mapping[key]
            try:
                self.model = joblib.load(model_file)
                self.scaler = scaler_mapping[self.scaler_choice]()
            except Exception as e:
                return

    def preprocess_data(self, test_data):
        """
        Preprocess the test data for classification.
        """
        processed_data = test_data.copy()

        if 'label' not in processed_data.columns:
            return None, None, "Error: Invalid dataset format. Ensure the 'label' column has valid values."

        processed_data['label'] = processed_data['label'].map(LABEL_MAPPING)

        numeric_cols = processed_data.select_dtypes(include=['number']).columns.tolist()
        if 'label' in numeric_cols:
            numeric_cols.remove('label')

        if processed_data['label'].isna().sum() > 0 or len(numeric_cols) == 0:
            return None, None, "Error: Invalid dataset format. Ensure the 'label' column has valid values."

        processed_data[numeric_cols] = self.scaler.fit_transform(processed_data[numeric_cols])

        X_test = processed_data.drop(columns=['label']).values
        y_test = processed_data['label'].values
        return X_test, y_test, None

    def classify(self, df_test):
        X_test, y_test, error_message = self.preprocess_data(df_test)

        if error_message:
            st.error(error_message)
            return

        y_pred = self.model.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        self.display_results(metrics)

    def display_results(self, metrics):
        st.write("### 📉 Performance Analysis")
        st.table(pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Value": [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']]
        }))

        st.write("### 🔢 Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=LABEL_MAPPING.keys(), yticklabels=LABEL_MAPPING.keys())
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        st.pyplot(fig)