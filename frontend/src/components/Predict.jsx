import React, { Link,useState } from 'react';
import api from '../api'; // Updated to use axios directly
import '../styles/predict.css';


const PredictionForm = () => {
  const [age, setAge] = useState(30);
  const [gender, setGender] = useState(0);
  const [pulse, setPulse] = useState(80);
  const [systolicBloodPressure, setSystolicBloodPressure] = useState(120);
  const [diastolicBloodPressure, setDiastolicBloodPressure] = useState(80);
  const [respiratoryRate, setRespiratoryRate] = useState(15);
  const [spo2, setSpo2] = useState(95);
  const [randomBloodSugar, setRandomBloodSugar] = useState(100);
  const [temperature, setTemperature] = useState(98.6);
  const [result, setResult] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
  
    try {
      const response = await api.post('/api/predictions/', {
        age,
        gender,
        pulse,
        systolicBloodPressure,
        diastolicBloodPressure,
        respiratoryRate,
        spo2,
        randomBloodSugar,
        temperature,
      }, {
        headers: {
          'Content-Type': 'application/json',
        },
      });
  
      setResult(response.data.prediction); // Assuming your API response structure
    } catch (error) {
      setError('Error fetching predictions. Please try again later.');
      console.error('Error fetching ML results:', error.response ? error.response.data : error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    
    
    <div className="login-root">
      <div className="box-root flex-flex flex-direction--column" style={{ minHeight: '60vh', flexGrow: 1 }}>
        <div className="loginbackground box-background--white padding-top--34">
          {/* Background styles */}
        </div>
        <div className="box-root padding-top--24 flex-flex flex-direction--column" style={{ flexGrow: 1, zIndex: 9 }}>
          <div className="box-root padding-top--28 padding-bottom--24 flex-flex flex-justifyContent--center">
            <h1><a href="#" rel="dofollow">Patient Classification</a></h1>
          </div>
          <div className="formbg-outer">
            <div className="formbg">
              <div className="formbg-inner padding-horizontal--48">
                
                <form onSubmit={handleSubmit} className="form-container">
                  <div className="field padding-bottom--14">
                    <label htmlFor="age">Age: {age}</label>
                    <input
                      type="range"
                      id="age"
                      name="age"
                      min="10"
                      max="100"
                      step="1"
                      value={age}
                      onChange={(e) => setAge(e.target.valueAsNumber)}
                      required
                    />
                  </div>
                  <div className="field padding-bottom--24">
                    <label htmlFor="gender">Gender:</label>
                    <select
                      id="gender"
                      name="gender"
                      value={gender}
                      onChange={(e) => setGender(parseInt(e.target.value))}
                      required
                    >
                      <option value="0">Female</option>
                      <option value="1">Male</option>
                    </select>
                  </div>
                  <div className="field padding-bottom--24">
                    <label htmlFor="pulse">Pulse: {pulse}</label>
                    <input
                      type="range"
                      id="pulse"
                      name="pulse"
                      min="35"
                      max="150"
                      step="1"
                      value={pulse}
                      onChange={(e) => setPulse(e.target.valueAsNumber)}
                      required
                    />
                  </div>
                  <div className="field padding-bottom--24">
                    <label htmlFor="systolicBloodPressure">Systolic Blood Pressure: {systolicBloodPressure}</label>
                    <input
                      type="range"
                      id="systolicBloodPressure"
                      name="systolicBloodPressure"
                      min="80"
                      max="200"
                      step="1"
                      value={systolicBloodPressure}
                      onChange={(e) => setSystolicBloodPressure(e.target.valueAsNumber)}
                      required
                    />
                  </div>
                  <div className="field padding-bottom--24">
                    <label htmlFor="diastolicBloodPressure">Diastolic Blood Pressure: {diastolicBloodPressure}</label>
                    <input
                      type="range"
                      id="diastolicBloodPressure"
                      name="diastolicBloodPressure"
                      min="50"
                      max="180"
                      step="1"
                      value={diastolicBloodPressure}
                      onChange={(e) => setDiastolicBloodPressure(e.target.valueAsNumber)}
                      required
                    />
                  </div>
                  <div className="field padding-bottom--24">
                    <label htmlFor="respiratoryRate">Respiratory Rate: {respiratoryRate}</label>
                    <input
                      type="range"
                      id="respiratoryRate"
                      name="respiratoryRate"
                      min="8"
                      max="50"
                      step="1"
                      value={respiratoryRate}
                      onChange={(e) => setRespiratoryRate(e.target.valueAsNumber)}
                      required
                    />
                  </div>
                  <div className="field padding-bottom--24">
                    <label htmlFor="spo2">SPO2: {spo2}</label>
                    <input
                      type="range"
                      id="spo2"
                      name="spo2"
                      min="70"
                      max="99"
                      step="1"
                      value={spo2}
                      onChange={(e) => setSpo2(e.target.valueAsNumber)}
                      required
                    />
                  </div>
                  <div className="field padding-bottom--24">
                    <label htmlFor="randomBloodSugar">Random Blood Sugar: {randomBloodSugar}</label>
                    <input
                      type="range"
                      id="randomBloodSugar"
                      name="randomBloodSugar"
                      min="80"
                      max="300"
                      step="1"
                      value={randomBloodSugar}
                      onChange={(e) => setRandomBloodSugar(e.target.valueAsNumber)}
                      required
                    />
                  </div>
                  <div className="field padding-bottom--24">
                    <label htmlFor="temperature">Temperature: {temperature}</label>
                    <input
                      type="range"
                      id="temperature"
                      name="temperature"
                      min="95"
                      max="108"
                      step="1"
                      value={temperature}
                      onChange={(e) => setTemperature(e.target.valueAsNumber)}
                      required
                    />
                  </div>
                  <div className="field padding-bottom--24">
                    <input
                      type="submit"
                      name="submit"
                      value="Predict"
                      className="btn btn-block btn-primary text-white py-3 px-5"
                      style={{ backgroundColor: '#CCBB00', borderColor: '#CCBB00' }}
                    />
                  </div>
                  <div className="field padding-bottom--24">
                    <h3 style={{ color: '#CCBB00' }}>Emergency: {result}</h3>
                    {error && <p className="error">{error}</p>}
                    {loading && <p>Loading...</p>}
                  </div>
                </form>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    
  );
};

export default PredictionForm;
