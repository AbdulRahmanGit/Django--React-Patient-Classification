import streamlit as st
import pandas as pd
import pickle
from model import (
    process_randomForest,
    process_decisionTree,
    process_naiveBayes,
    process_knn,
    process_LogisticRegression,
    process_SVM
)

# Specify the path to your CSV file
csv_file_path = 'EmergencyDataset.csv'  # Update this path

# Load the dataset
try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    st.error("The specified CSV file was not found. Please check the file path.")
    df = pd.DataFrame()  # Create an empty DataFrame if the file is not found

# Sidebar navigation
pages = {
    "Home": "home",
    "Data": "data",
    "Classification Results": "results",
    "Predict": "predict"
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))

if selection == "Data":
    st.title("Dataset")
    if not df.empty:
        st.write(df)

        # Download button for the CSV file
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='EmergencyDataset.csv',
            mime='text/csv',
        )
    else:
        st.write("No data available to display.")

elif selection == "Classification Results":
    st.title("Classification Results")

    # Call classification functions and collect results
    models = {
        "Decision Tree": process_decisionTree(),  # Ensure this returns (model, report)
        "Random Forest": process_randomForest(),
        "Logistic Regression": process_LogisticRegression(),
        "Support Vector Machine": process_SVM(),
        "Naive Bayes": process_naiveBayes(),
        "KNN": process_knn()
    }

    # Prepare a DataFrame to hold the results
    results = {
        "Model": [],
        "Metric": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": [],
        "Support": []
    }

    for model_name, (model, report) in models.items():  # Unpack model and report
        for key in report.keys():
            if key not in ['accuracy', 'macro avg', 'weighted avg']:
                results["Model"].append(model_name)
                results["Metric"].append(key)
                results["Precision"].append(report[key]['precision'])
                results["Recall"].append(report[key]['recall'])
                results["F1 Score"].append(report[key]['f1-score'])
                results["Support"].append(report[key]['support'])

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    # Display the results in a table format
    st.dataframe(results_df)

    # Determine the best model based on accuracy
    best_model_name = max(models, key=lambda name: models[name][1]['accuracy'])  # Assuming 'accuracy' is in the report
    best_model = models[best_model_name][0]  # Get the best model directly
    st.write(f"The best model is: {best_model_name}")

elif selection == "Predict":
    st.title("Predict Patient Classification")
    # Input fields for prediction with default values
    age = st.number_input("Age", min_value=1, max_value=120, step=1, format="%d", value=30)  # Default age is 30
    gender = st.selectbox("Gender", ["Male", "Female"], index=0)  # Default gender is Male
    pulse = st.number_input("Pulse", min_value=30, max_value=200, step=1, format="%d", value=70)  # Default pulse is 70 bpm
    systolic_bp = st.number_input("Systolic Blood Pressure", min_value=70, max_value=300, step=1, format="%d", value=120)  # Default systolic BP is 120 mmHg
    diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=200, step=1, format="%d", value=80)  # Default diastolic BP is 80 mmHg
    respiratory_rate = st.number_input("Respiratory Rate", min_value=1, max_value=100, step=1, format="%d", value=16)  # Default respiratory rate is 16 breaths/min
    spo2 = st.number_input("SPO2", min_value=80, max_value=100, step=1, format="%d", value=98)  # Default SPO2 is 98%
    random_blood_sugar = st.number_input("Random Blood Sugar", min_value=0, max_value=400, step=1, format="%d", value=100)  # Default random blood sugar is 100 mg/dL
    temperature = st.number_input("Temperature (°F)", min_value=95, max_value=104, step=1, format="%d", value=98)  # Default temperature is 98°F

    # Validate inputs
    if st.button("Submit"):
        if age < 1 or age > 120:
            st.error("Age must be between 1 and 120.")
        elif pulse < 30 or pulse > 200:
            st.error("Pulse must be between 30 and 200 bpm.")
        elif systolic_bp < 70 or systolic_bp > 300:
            st.error("Systolic Blood Pressure must be between 70 and 300 mmHg.")
        elif diastolic_bp < 40 or diastolic_bp > 200:
            st.error("Diastolic Blood Pressure must be between 40 and 200 mmHg.")
        elif respiratory_rate < 1 or respiratory_rate > 100:
            st.error("Respiratory Rate must be between 1 and 100 breaths/min.")
        elif spo2 < 80 or spo2 > 100:
            st.error("SPO2 must be between 80% and 100%.")
        elif random_blood_sugar < 0 or random_blood_sugar > 400:
            st.error("Random Blood Sugar must be between 0 and 400 mg/dL.")
        elif temperature < 95 or temperature > 104:
            st.error("Temperature must be between 95°F and 104°F.")
        else:
            # Prepare the input data for prediction with correct feature names
            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [1 if gender == 'Male' else 0],  # Assuming binary encoding for gender
                'Pulse': [pulse],
                'SystolicBloodPressure': [systolic_bp],  # Corrected feature name
                'DiastolicBloodPressure': [diastolic_bp],  # Corrected feature name
                'RespiratoryRate': [respiratory_rate],  # Corrected feature name
                'SPO2': [spo2],
                'RandomBloodSugar': [random_blood_sugar],  # Corrected feature name
                'Temperature': [temperature]
            })

            # Load the models from the pickle file
            try:
                with open('all_models.pkl', 'rb') as f:
                    models = pickle.load(f)
            except FileNotFoundError:
                st.error("Model file not found. Please ensure 'all_models.pkl' is in the correct directory.")
                st.stop()  # Use st.stop() to halt execution in Streamlit

            # Select the best model based on evaluation metrics (e.g., accuracy or TP values)
            best_model_name = max(models, key=lambda name: models[name][1]['accuracy'])  # Get the name of the best model
            best_model_info = models[best_model_name]  # Get the tuple (model, report)
            best_model = best_model_info[0]  # Access the model from the tuple

            # Now you can make predictions
            prediction = best_model.predict(input_data)[0]
            if prediction == 0:
                st.write(f"The Emergency Level is: Not Critical")
            elif prediction == 1:
                st.write(f"The Emergency Level is: Critical")

# ... existing code for other pages ...
elif selection == "Home":
    # Home page content
    st.title("Patient Categorization")

    # Introduction section
    st.header("MACHINE LEARNING BASED PATIENT CLASSIFICATION IN EMERGENCY DEPARTMENT")
    st.write(
        "This work contains the classification of patients in an Emergency Department in a hospital according to their critical conditions. "
        "Machine learning can be applied based on the patient's condition to quickly determine if the patient requires urgent medical intervention from the clinicians."
    )

    # Objective Evaluation section
    st.header("Objective Evaluation")
    st.write("Our analysis of the dataset revealed several key insights:")
    st.markdown(
        """
        - High accuracy rates in classifying patients based on vital signs such as Systolic Blood Pressure (SBP) and Diastolic Blood Pressure (DBP).
        - Machine learning models demonstrated the ability to predict critical conditions with a precision of over 90%.
        - Random Forest and Support Vector Machine models outperformed traditional methods in terms of recall and F1 score.
        - Feature importance analysis indicated that age and vital signs were the most significant predictors of patient outcomes.
        """
    )

    # View Published Paper section
    st.header("View Published Paper")
    st.write("For a detailed understanding of our methodology and findings, you can read the full paper:")
    st.markdown("[Read the Full Paper](https://drive.google.com/file/d/12nlamj2-fqe0g2JZRfXTEMzYw95QaUVP/view?usp=sharing)")  # Update the path to your published paper

    # Footer
    st.write("© 2024 Patient Categorization Project. All rights reserved.")