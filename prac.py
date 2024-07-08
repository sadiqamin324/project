import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

# Load Data
@st.cache_data
def load_data():
    try:
        cust = pd.read_csv("https://raw.githubusercontent.com/sadiqamin324/project/main/cust.csv")
        trans = pd.read_csv("https://raw.githubusercontent.com/sadiqamin324/project/main/trans.csv")
        
        merged_df = pd.merge(cust, trans, on='customerEmail', how='inner')
        merged_df = merged_df.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y'])
        return merged_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Streamlit App
st.title('Fraud Email Detection')

# Load the data from URLs directly
df = load_data()

if not df.empty:
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

    data = preprocess_data(df)

    # Encode Data
    def encode_data(data):
        features = ['No_Transactions', 'No_Orders', 'No_Payments', 'Transaction_Success_Rate', 'Transaction_TotalAmount']
        X = data[features]
        y = data['Fraud']
        return X, y

    X, y = encode_data(data)

    # Train Models with Ensemble Methods
    def train_models(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Improved Random Forest Model with Hyperparameter Tuning
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }
        rf_model = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, n_iter=50, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        
        # Gradient Boosting Model
        gb_model = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
        gb_model.fit(X_train, y_train)

        return rf_model.best_estimator_, gb_model, X_test, y_test

    rf_model, gb_model, X_test, y_test = train_models(X, y)

    # Evaluate Models
    def evaluate_model(y_test, predictions):
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        return accuracy, precision, recall, f1, roc_auc, cm

    rf_predictions = rf_model.predict(X_test)
    gb_predictions = gb_model.predict(X_test)

    rf_metrics = evaluate_model(y_test, rf_predictions)
    gb_metrics = evaluate_model(y_test, gb_predictions)

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
            model_option = st.selectbox("Choose Model", ["RandomForest", "GradientBoosting"])
            if model_option == "RandomForest":
                metrics = rf_metrics
                model = rf_model
            else:
                metrics = gb_metrics
                model = gb_model

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
            prediction = model.predict(input_features)
            st.write(f"Prediction: {'Fraud' if prediction[0] == 1 else 'Not Fraud'}")
        else:
            st.write("No data found for this email.")
else:
    st.write("Error loading data.")
