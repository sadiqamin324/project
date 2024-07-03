import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

# Load Data
@st.cache_data
def load_data():
    try:
        cust = pd.read_csv(r"https://raw.githubusercontent.com/sadiqamin324/project/main/cust.csv")
        trans = pd.read_csv(r"https://raw.githubusercontent.com/sadiqamin324/project/main/trans.csv")
        merged_df = pd.merge(cust, trans, on='customerEmail', how='inner')
        merged_df = merged_df.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y'])
        return merged_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

# Data Preprocessing
def preprocess_data(df):
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    Q1 = df[numerical_features].quantile(0.25)
    Q3 = df[numerical_features].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (df[numerical_features] < lower_bound) | (df[numerical_features] > upper_bound)
    df_cleaned = df[~outliers.any(axis=1)]
    df_cleaned['Transaction_Success_Rate'] = ((df_cleaned['No_Transactions'] / (df_cleaned['No_Transactions'] + df_cleaned['transactionFailed'])) * 100).round(2)
    df_cleaned['Transaction_TotalAmount'] = df_cleaned['No_Transactions'] * df_cleaned['transactionAmount']
    
    # Impute missing values for all columns
    imputer = SimpleImputer(strategy='most_frequent')
    df_cleaned = pd.DataFrame(imputer.fit_transform(df_cleaned), columns=df_cleaned.columns)
    
    # Ensure 'Fraud' column is binary
    df_cleaned['Fraud'] = df_cleaned['Fraud'].astype(int)
    
    return df_cleaned

if not df.empty:
    data = preprocess_data(df)
else:
    data = pd.DataFrame()

# Encode Data
def encode_data(data):
    features = ['No_Transactions', 'No_Orders', 'No_Payments', 'Transaction_Success_Rate', 'Transaction_TotalAmount']
    X = data[features]
    y = data['Fraud']
    return X, y

if not data.empty:
    X, y = encode_data(data)
else:
    X, y = pd.DataFrame(), pd.Series()

# Train Models
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # SVM
    svm_model = SVC(random_state=42, probability=True)
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)

    # RandomForest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    return svm_model, rf_model, X_test, y_test, svm_predictions, rf_predictions

if not X.empty and not y.empty:
    svm_model, rf_model, X_test, y_test, svm_predictions, rf_predictions = train_models(X, y)
else:
    svm_model, rf_model, X_test, y_test, svm_predictions, rf_predictions = None, None, None, None, None, None

# Evaluate Models
def evaluate_model(y_test, predictions, model):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    cm = confusion_matrix(y_test, predictions)
    return accuracy, precision, recall, f1, roc_auc, cm

if svm_model and rf_model:
    svm_metrics = evaluate_model(y_test, svm_predictions, svm_model)
    rf_metrics = evaluate_model(y_test, rf_predictions, rf_model)
else:
    svm_metrics, rf_metrics = None, None

# Streamlit App
st.title('Fraud Email Detection')

email = st.text_input("Enter the customer's email")

if email:
    customer_data = data[data['customerEmail'] == email]

    if not customer_data.empty:
        st.write("## Customer Data")
        st.write(customer_data)

        st.write("### Numerical Features")
        numerical_features = ['No_Transactions', 'No_Orders', 'No_Payments', 'Transaction_Success_Rate', 'Transaction_TotalAmount']
        for feature in numerical_features:
            st.write(f"{feature}: {customer_data[feature].values[0]}")

        st.write("### Data Visualization")
        viz_option = st.selectbox("Choose Visualization", ["Histograms", "Boxplots", "Correlation Matrix"])
        if viz_option == "Histograms":
            st.write("### Histograms of Numerical Features")
            fig, ax = plt.subplots(1, len(numerical_features), figsize=(15, 5))
            for i, feature in enumerate(numerical_features):
                sns.histplot(data[feature], ax=ax[i], kde=True)
            st.pyplot(fig)
        elif viz_option == "Boxplots":
            st.write("### Boxplots of Numerical Features")
            fig, ax = plt.subplots(1, len(numerical_features), figsize=(15, 5))
            for i, feature in enumerate(numerical_features):
                sns.boxplot(y=data[feature], ax=ax[i])
            st.pyplot(fig)
        elif viz_option == "Correlation Matrix":
            st.write("### Correlation Matrix")
            correlation_matrix = data[numerical_features + ['Fraud']].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            st.pyplot(fig)

        st.write("### Model Evaluation")
        model_option = st.selectbox("Choose Model", ["SVM", "RandomForest"])
        if model_option == "SVM":
            metrics = svm_metrics
        else:
            metrics = rf_metrics

        if metrics:
            st.write(f"### {model_option} Metrics")
            st.write(f"Accuracy: {metrics[0]:.2f}")
            st.write(f"Precision: {metrics[1]:.2f}")
            st.write(f"Recall: {metrics[2]:.2f}")
            st.write(f"F1 Score: {metrics[3]:.2f}")
            st.write(f"ROC AUC: {metrics[4]:.2f}")
            st.write("Confusion Matrix:")
            st.write(metrics[5])

        st.write("### Predict Fraud")
        input_features = customer_data[['No_Transactions', 'No_Orders', 'No_Payments', 'Transaction_Success_Rate', 'Transaction_TotalAmount']]
        if model_option == "SVM":
            prediction = svm_model.predict(input_features)
        else:
            prediction = rf_model.predict(input_features)
        st.write(f"Prediction: {'Fraud' if prediction[0] == 1 else 'Not Fraud'}")
    else:
        st.write("No data found for this email.")
