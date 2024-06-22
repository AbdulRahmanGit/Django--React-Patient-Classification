import React from 'react';
import { Link } from 'react-router-dom';

const UserHome = () => (
  <><nav>
  <a href="#">Patient Classification</a>
  <ul className="list">
    <li><Link to="/">Home</Link></li>
    <li><Link to="/Dataset">Dataset</Link></li>
    <li><Link to="/MLResults">Classification</Link></li>
    <li><Link to="/Predict">Predict</Link></li>
    <li><Link to="/Logout">Log out</Link></li>
  </ul>
</nav>
    <header id="header_wrapper">
      <div className="container">
        <div className="header_box">
          <div className="logo">
            <Link to="/"><h2>Patient Categorization</h2></Link>
          </div>
          <nav className="navbar navbar-inverse" role="navigation">
            <div className="navbar-header">
              <button
                type="button"
                id="nav-toggle"
                className="navbar-toggle"
                data-toggle="collapse"
                data-target="#main-nav"
              >
                <span className="sr-only">Toggle navigation</span>
                <span className="icon-bar"></span>
                <span className="icon-bar"></span>
                <span className="icon-bar"></span>
              </button>
            </div>
            <div id="main-nav" className="collapse navbar-collapse navStyle">
              <ul className="nav navbar-nav" id="mainNav">
                <li><Link to="/">Home</Link></li>
                <li><Link to="/predict">Predict</Link></li>
                <li><Link to="/MLResults">Classification</Link></li>
                <li><Link to="/view-dataset">Dataset</Link></li>
                <li><Link to="/logout">Logout</Link></li>
              </ul>
            </div>
          </nav>
        </div>
      </div>
    </header>

    <section id="hero_section" className="top_cont_outer">
      <div className="hero_wrapper">
        <div className="container">
          <div className="hero_section">
            <div className="row">
              <div className="col-lg-5 col-sm-7">
                <div className="top_left_cont zoomIn wow animated">
                  <h2>
                    <strong>Automated Patient Categorization in the Emergency Room Using Machine Learning</strong>
                  </h2>
                  <p>
                    This work contains the classification of patients in an Emergency Department in a hospital according to their critical conditions. Machine learning can be applied based on the patient's condition to quickly determine if the patient requires urgent medical intervention from the clinicians or not. Basic vital signs like Systolic Blood Pressure (SBP), Diastolic Blood Pressure (DBP), Respiratory Rate (RR), Oxygen saturation (SPO2), Random Blood Sugar (RBS), Temperature, Pulse Rate (PR) are used as the input for the patient's risk level identification. High-risk or non-risk categories are considered as the output for patient classification. Basic machine learning techniques such as LR, Gaussian NB, SVM, KNN, and DT are used for the classification. Precision, recall, and F1-score are considered for the evaluation. The decision tree gives best F1-score of 77.67 for the risk level classification of the imbalanced dataset.
                  </p>
                  <Link to="https://docs.google.com/document/d/1dS3AytpBF_vbUxNZAyLamBkzbUXMLu-dm0siMgxR51s/edit?usp=sharing" className="btn btn-dark">Read more</Link>
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

    <footer className="footer_wrapper" id="contact">
      <div className="container">
        <div className="footer_bottom">
          <span>Template by Your Name</span>
        </div>
      </div>
    </footer>
  </>
);

export default UserHome;
