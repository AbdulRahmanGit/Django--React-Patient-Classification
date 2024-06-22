import React, { useState, useEffect } from 'react';
import api from '../api';
import '../styles/view.css';

const CsvViewer = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await api.get('/api/view-dataset/');
        setData(response.data);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching dataset:', error);
        setError('Failed to fetch dataset');
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Get headers from the first row of the data
  const headers = data.length > 0 ? Object.keys(data[0]) : [];

  return (
    <div className="csv-viewer">
      
      <h2>CSV Viewer</h2>
      {loading ? (
        <p>Loading dataset...</p>
      ) : error ? (
        <p>{error}</p>
      ) : (
        <div className="csv-viewer-table">
          <table className="table">
            <thead>
              <tr>
                {headers.map((header, index) => (
                  <th key={index}>{header}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map((row, index) => (
                <tr key={index}>
                  {headers.map((header, colIndex) => (
                    <td key={colIndex}>{row[header]}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default CsvViewer;
