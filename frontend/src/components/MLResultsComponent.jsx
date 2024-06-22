import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom'; // Import Link from react-router-dom
import api from '../api';
import '../styles/mlresults.css';
import '../styles/Home.css';
const MLResultsComponent = () => {
  const [mlResults, setMLResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchMLResults = async () => {
      try {
        const response = await api.get('/api/userClassificationResults');
        setMLResults(response.data);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching ML results:', error);
        setError('Failed to fetch ML results');
        setLoading(false);
      }
    };

    fetchMLResults();
  }, []);

  const renderTable = (data, title) => {
    if (!data || Object.keys(data).length === 0) {
      return null; // Return null or handle empty data case
    }

    return (
      <div className="ml-results-table">
        <h3>{title}</h3>
        <table className="table">
          <thead>
            <tr>
              {Object.keys(data[Object.keys(data)[0]]).map((key) => (
                <th key={key}>{key}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.keys(data).map((rowKey) => (
              <tr key={rowKey}>
                {Object.values(data[rowKey]).map((value, idx) => (
                  <td key={idx}>{value}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  return (
    <div className="home-container">
    <nav className="navbar">
      <div className="navbar-header">
        <a href="#" className="logo">Patient Classification</a>
      </div>
      <ul className="navbar-nav">
        <li className="nav-item"><Link to="/">Home</Link></li>
        <li className="nav-item"><Link to="/Dataset">Dataset</Link></li>
        <li className="nav-item"><Link to="/MLResults">Classification</Link></li>
        <li className="nav-item"><Link to="/Predict">Predict</Link></li>
        <li className="nav-item"><Link to="/Logout">Log out</Link></li>
      </ul>
    </nav>

      <div className="ml-results">
        <h2>Machine Learning Results</h2>
        {loading ? (
          <p>Loading ML results...</p>
        ) : error ? (
          <p>{error}</p>
        ) : (
          <div className="ml-results-tables">
            {renderTable(mlResults?.lg, 'Logistic Regression')}
            {renderTable(mlResults?.svm, 'Support Vector Machine')}
            {renderTable(mlResults?.rf, 'Random Forest')}
            {renderTable(mlResults?.dt, 'Decision Tree')}
            {renderTable(mlResults?.nb, 'Naive Bayes')}
            {renderTable(mlResults?.gb, 'Gradient Boosting')}
          </div>
        )}
      </div>
    </div>
  );
};

export default MLResultsComponent;
