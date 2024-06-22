import React from 'react';
import { Link } from 'react-router-dom';

const Home = () => (
  <section id="hero_section" className="top_cont_outer">
    <div className="hero_wrapper">
      <div className="container">
        <div className="hero_section">
          <div className="row">
            <div className="col-lg-5 col-sm-7">
              <div className="top_left_cont zoomIn wow animated">
                <h2><strong>Automated Patient Categorization in the Emergency Room Using Machine Learning</strong></h2>
                <p>
                  This work contains the classification of patients in an Emergency Department in a hospital according to their
                  critical conditions. Machine learning can be applied based on the patient's condition to quickly determine if the patient requires
                  urgent medical intervention from the clinicians or not. Basic vital signs like Systolic Blood Pressure (SBP), Diastolic Blood Pressure
                  (DBP), Respiratory Rate (RR), Oxygen saturation (SPO2), Random Blood Sugar (RBS), Temperature, Pulse Rate (PR) are used
                  as the input for the patient's risk level identification. High-risk or non-risk categories are considered as the output for patient
                  classification. Basic machine learning techniques such as LR, Gaussian NB, SVM, KNN and DT are used for the classification.
                  Precision, recall, and F1-score are considered for the evaluation. The decision tree gives best F1-score of 77.67 for the risk level
                  classification of the imbalanced dataset.
                </p>
                {/* <a href="#service" className="read_more2">Read more</a> */}
              </div>
            </div>
            <div className="col-lg-7 col-sm-5">
              {/* <img src="/static/img/portfolio_pic7.jpg" className="zoomIn wow animated" alt="" /> */}
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
);

export default Home;
