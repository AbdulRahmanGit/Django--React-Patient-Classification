// src/components/Home.js
import React from 'react';
import { Link } from 'react-router-dom';
import '../styles/Home.css'; // Adjust import path if necessary

const Home = () => (
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
    <div className="content-container">
      <div className="background-animation">
        <div className="box-root box-background--blue800"></div>
        <div className="box-root box-background--blue animationLeftRight"></div>
        <div className="box-root box-background--blue animationRightLeft"></div>
      </div>
      <div className="main-content">
        <h2>Welcome to Your Dashboard</h2>
        <p>Explore your patient Data classification and machine learning results.</p>
        <Link to="/Predict" className="btn">Predict</Link>
        <a href='https://docs.google.com/document/d/1dS3AytpBF_vbUxNZAyLamBkzbUXMLu-dm0siMgxR51s/edit?usp=sharing' className='btn'>Read Publish Paper</a>
      </div>
    </div>
    
    
  </div>
);

export default Home;
