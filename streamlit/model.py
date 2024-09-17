import pandas as pd
import os  # Unused import, can be removed
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle
import logging
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load dataset
path = os.path.join('streamlit', 'EmergencyDataset.csv')
df = pd.read_csv(path)
X = df.iloc[:, :-1].values  # Keep as DataFrame for feature names
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=109)

@st.cache_data
def process_randomForest():
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    rf_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    logging.info(f"Classification report for Random Forest:\n{rf_report}\n")
    
    return clf, rf_report  # Ensure this returns both the model and the report

@st.cache_data
def process_decisionTree():
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    dt_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    logging.info(f"Classification report for Decision Tree:\n{dt_report}\n")
    
    return clf, dt_report  # Return both the model and the report

@st.cache_data
def process_naiveBayes():
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    nb_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    logging.info(f"Classification report for Naive Bayes:\n{nb_report}\n")
    
    return clf, nb_report  # Return both the model and the report

@st.cache_data
def process_knn():
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    knn_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    logging.info(f"Classification report for KNN:\n{knn_report}\n")
    
    return clf, knn_report  # Return both the model and the report

@st.cache_data
def process_LogisticRegression():
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(solver='liblinear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    lg_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    logging.info(f"Classification report for Logistic Regression:\n{lg_report}\n")
    
    return clf, lg_report  # Return both the model and the report

@st.cache_data
def process_SVM():
    from sklearn.svm import SVC
    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    svc_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    logging.info(f"Classification report for SVM:\n{svc_report}\n")
    
    return clf, svc_report  # Return both the model and the report

def save_all_models(models):
    """Save all models to a single pickle file."""
    model_file = 'all_models.pkl'  # Save in the main root directory
    try:  # Added error handling
        with open(model_file, 'wb') as f:
            pickle.dump(models, f)
        logging.info(f"All models saved to {model_file}")
    except Exception as e:
        logging.error(f"Error saving models: {e}")

def train_and_save_models():
    """Train all models and save them to a single file."""
    models = {}
    
    # Train each model and store both the model and the report in the dictionary
    models['Random Forest'] = process_randomForest()  # Store both model and report
    models['Decision Tree'] = process_decisionTree()
    models['Naive Bayes'] = process_naiveBayes()
    models['KNN'] = process_knn()
    models['Logistic Regression'] = process_LogisticRegression()
    models['SVM'] = process_SVM()
    
    # Save all models to a single file
    save_all_models(models)

# Call this function to train and save all models
train_and_save_models()