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
import random
import os
# Specify the path to your CSV file
csv_file_path = os.path.join('EmergencyDataset.csv')

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
    
    # Function to generate random values
    def generate_random_values():
        return {
            'age': random.randint(1, 100),
            'gender': random.choice(["Male", "Female"]),
            'pulse': random.randint(30, 150),
            'systolic_bp': random.randint(70, 200),
            'diastolic_bp': random.randint(40, 200),
            'respiratory_rate': random.randint(1, 100),
            'spo2': random.randint(80, 100),
            'random_blood_sugar': random.randint(1, 400),
            'temperature': random.randint(95, 104)
        }

    # Initialize session state for input values if not already done
    if 'age' not in st.session_state:
        st.session_state.age = 30
    if 'gender' not in st.session_state:
        st.session_state.gender = "Male"
    if 'pulse' not in st.session_state:
        st.session_state.pulse = 70
    if 'systolic_bp' not in st.session_state:
        st.session_state.systolic_bp = 120
    if 'diastolic_bp' not in st.session_state:
        st.session_state.diastolic_bp = 80
    if 'respiratory_rate' not in st.session_state:
        st.session_state.respiratory_rate = 16
    if 'spo2' not in st.session_state:
        st.session_state.spo2 = 98
    if 'random_blood_sugar' not in st.session_state:
        st.session_state.random_blood_sugar = 100
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 98

    # Input fields for prediction with values from session state
    st.session_state.age = st.number_input("Age", min_value=1, max_value=120, step=1, format="%d", value=st.session_state.age)
    st.session_state.gender = st.selectbox("Gender", ["Male", "Female"], index=0 if st.session_state.gender == "Male" else 1)
    st.session_state.pulse = st.number_input("Pulse", min_value=30, max_value=200, step=1, format="%d", value=st.session_state.pulse)
    st.session_state.systolic_bp = st.number_input("Systolic Blood Pressure", min_value=70, max_value=300, step=1, format="%d", value=st.session_state.systolic_bp)
    st.session_state.diastolic_bp = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=200, step=1, format="%d", value=st.session_state.diastolic_bp)
    st.session_state.respiratory_rate = st.number_input("Respiratory Rate", min_value=1, max_value=100, step=1, format="%d", value=st.session_state.respiratory_rate)
    st.session_state.spo2 = st.number_input("SPO2", min_value=80, max_value=100, step=1, format="%d", value=st.session_state.spo2)
    st.session_state.random_blood_sugar = st.number_input("Random Blood Sugar", min_value=0, max_value=400, step=1, format="%d", value=st.session_state.random_blood_sugar)
    st.session_state.temperature = st.number_input("Temperature (°F)", min_value=95, max_value=104, step=1, format="%d", value=st.session_state.temperature)

    # Button to generate random values
    if st.button("Generate Random Values"):
        random_values = generate_random_values()
        st.session_state.age = random_values['age']
        st.session_state.gender = random_values['gender']
        st.session_state.pulse = random_values['pulse']
        st.session_state.systolic_bp = random_values['systolic_bp']
        st.session_state.diastolic_bp = random_values['diastolic_bp']
        st.session_state.respiratory_rate = random_values['respiratory_rate']
        st.session_state.spo2 = random_values['spo2']
        st.session_state.random_blood_sugar = random_values['random_blood_sugar']
        st.session_state.temperature = random_values['temperature']
        
        # Display the generated random values
        st.success("Random values generated!")
        st.write(f"Age: {st.session_state.age}, Gender: {st.session_state.gender}, Pulse: {st.session_state.pulse}, Systolic BP: {st.session_state.systolic_bp}, Diastolic BP: {st.session_state.diastolic_bp}, Respiratory Rate: {st.session_state.respiratory_rate}, SPO2: {st.session_state.spo2}, Random Blood Sugar: {st.session_state.random_blood_sugar}, Temperature: {st.session_state.temperature}")

    # Validate inputs
    if st.button("Submit"):
        if st.session_state.age < 1 or st.session_state.age > 120:
            st.error("Age must be between 1 and 120.")
        elif st.session_state.pulse < 30 or st.session_state.pulse > 200:
            st.error("Pulse must be between 30 and 200 bpm.")
        elif st.session_state.systolic_bp < 70 or st.session_state.systolic_bp > 300:
            st.error("Systolic Blood Pressure must be between 70 and 300 mmHg.")
        elif st.session_state.diastolic_bp < 40 or st.session_state.diastolic_bp > 200:
            st.error("Diastolic Blood Pressure must be between 40 and 200 mmHg.")
        elif st.session_state.respiratory_rate < 1 or st.session_state.respiratory_rate > 100:
            st.error("Respiratory Rate must be between 1 and 100 breaths/min.")
        elif st.session_state.spo2 < 80 or st.session_state.spo2 > 100:
            st.error("SPO2 must be between 80% and 100%.")
        elif st.session_state.random_blood_sugar < 0 or st.session_state.random_blood_sugar > 400:
            st.error("Random Blood Sugar must be between 0 and 400 mg/dL.")
        elif st.session_state.temperature < 95 or st.session_state.temperature > 104:
            st.error("Temperature must be between 95°F and 104°F.")
        else:
            # Prepare the input data for prediction with correct feature names
            input_data = pd.DataFrame({
                'Age': [st.session_state.age],
                'Gender': [1 if st.session_state.gender == 'Male' else 0],  # Assuming binary encoding for gender
                'Pulse': [st.session_state.pulse],
                'SystolicBloodPressure': [st.session_state.systolic_bp],  # Corrected feature name
                'DiastolicBloodPressure': [st.session_state.diastolic_bp],  # Corrected feature name
                'RespiratoryRate': [st.session_state.respiratory_rate],  # Corrected feature name
                'SPO2': [st.session_state.spo2],
                'RandomBloodSugar': [st.session_state.random_blood_sugar],  # Corrected feature name
                'Temperature': [st.session_state.temperature]
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
            input_data_np = input_data.to_numpy()
            # Now you can make predictions
            prediction = best_model.predict(input_data_np)[0]
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
