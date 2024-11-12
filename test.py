import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# Charger et préparer les données
@st.cache_data
def load_and_preprocess_data():
    try:
        df = pd.read_csv("data/Invistico_Airline.csv")  
    except FileNotFoundError:
        st.error("Le fichier 'Invistico_Airline.csv' n'a pas été trouvé. Veuillez vérifier le chemin.")
        return None, None, None, None, None, None, None, None, None
    
    df.columns = df.columns.str.strip()


    # st.write("Colonnes disponibles dans le DataFrame:")
    # st.write(df.columns)

    if 'satisfaction' in df.columns:
        df = pd.get_dummies(df, columns=['satisfaction'], drop_first=False)

    X = df.drop(columns=['satisfaction_dissatisfied', 'satisfaction_satisfied'])  
    y = df['satisfaction_satisfied']
    
    # Prétraitement des données
    num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = X.select_dtypes(exclude=['float64', 'int64']).columns.tolist()

    num_pipeline = Pipeline([ 
        ('imputer', SimpleImputer(strategy='mean')), 
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([ 
        ('imputer', SimpleImputer(strategy='most_frequent')),  
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combinaison des pipelines
    full_preprocess = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    # Appliquer le prétraitement
    X_processed = full_preprocess.fit_transform(X)
    
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, train_size=0.8, random_state=42)
    
    return df, full_preprocess, num_cols, cat_cols, X_train, X_test, y_train, y_test, X.columns

# Charger les modèles
@st.cache_resource
def load_models(X_train, y_train):
    model_linreg = LogisticRegression()
    model_linreg.fit(X_train, y_train)

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)

    model_nb = GaussianNB()
    model_nb.fit(X_train, y_train)
    
    return model_linreg, knn, model_nb

# Interface utilisateur Streamlit
def app():

    df, full_preprocess, num_cols, cat_cols, X_train, X_test, y_train, y_test, X_columns = load_and_preprocess_data()
    if df is None:
        return  
    
    model_linreg, knn, model_nb = load_models(X_train, y_train)


    st.title("Prédiction de la satisfaction client")
    st.write("Cette application utilise des modèles de machine learning pour prédire la satisfaction client.")
    

    user_input = {}
    for col in X_columns:
        if col in num_cols:
            user_input[col] = st.number_input(f"{col} ", value=0, key=col)
        elif col in cat_cols:
            user_input[col] = st.selectbox(f"{col} ", [0, 1], key=col)
    

    model_choice = st.selectbox("Choisissez un modèle", ("Régression Linéaire", "KNN", "Naïve Bayes"))


    if st.button("Prédire"):
        input_data = pd.DataFrame([user_input], columns=X_columns)

        processed_input_data = full_preprocess.transform(input_data)


        if model_choice == "Régression Linéaire":
            probas = model_linreg.predict_proba(processed_input_data)
            prediction = model_linreg.predict(processed_input_data)
            satisfaction_prediction = "satisfait" if prediction >= 0.5 else "non satisfait"
            prob_satisfaction = probas[0][1]  
        elif model_choice == "KNN":
            probas = knn.predict_proba(processed_input_data)
            prediction = knn.predict(processed_input_data)
            satisfaction_prediction = "satisfait" if prediction == 1 else "non satisfait"
            prob_satisfaction = probas[0][1]  
        elif model_choice == "Naïve Bayes":
            probas = model_nb.predict_proba(processed_input_data)
            prediction = model_nb.predict(processed_input_data)
            satisfaction_prediction = "satisfait" if prediction == 1 else "non satisfait"
            prob_satisfaction = probas[0][1]  

        st.subheader("Résultat de la prédiction")
        st.write(f"Le modèle prédit que le client est : **{satisfaction_prediction}**")
        st.write(f"Probabilité de satisfaction : **{prob_satisfaction:.2f}**")



    st.subheader("Évaluation des Modèles")
    model_eval = st.selectbox("Choisissez un modèle pour évaluation", ("Régression Linéaire", "KNN", "Naïve Bayes"))

    if st.button("Évaluer"):

        y_test = y_test.astype(int)

        if model_eval == "Régression Linéaire":
            y_test_pred = model_linreg.predict(X_test)
            y_test_pred = (y_test_pred >= 0.5).astype(int)  
            st.write("MSE:", mean_squared_error(y_test, y_test_pred))
            st.write("R²:", r2_score(y_test, y_test_pred))
        elif model_eval == "KNN":
            y_test_pred = knn.predict(X_test)
            y_test_pred = y_test_pred.astype(int)  
            st.write("Accuracy:", accuracy_score(y_test, y_test_pred))
            st.write("Classification Report:\n", classification_report(y_test, y_test_pred))
        elif model_eval == "Naïve Bayes":
            y_test_pred = model_nb.predict(X_test)
            y_test_pred = y_test_pred.astype(int)  
            st.write("Accuracy:", accuracy_score(y_test, y_test_pred))
            st.write("Classification Report:\n", classification_report(y_test, y_test_pred))



if __name__ == "__main__":
    app()
