import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import api from "../api";
import { ACCESS_TOKEN, REFRESH_TOKEN } from "../constants";
import '../styles/style.css';
import '../styles/Form1.css';
import '../styles/LoadingIndicator.css';

function Form({ route, method }) {
  const [formData, setFormData] = useState({ username: "", password: "", email: "" });
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      let response;
      if (method === "register") {
        response = await api.post(route, formData);
        if (response.status === 201) {
          navigate("/login");
        }
      } else {
        response = await api.post(route, formData);
        localStorage.setItem(ACCESS_TOKEN, response.data.access);
        localStorage.setItem(REFRESH_TOKEN, response.data.refresh);
        navigate("/");
      }
    } catch (err) {
      console.error("Error during form submission:", err);
      if (err.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        if (err.response.status === 400) {
          setError("Invalid input. Please check your details and try again.");
        } else if (err.response.status === 401) {
          setError("Authentication failed. Please check your credentials.");
        } else if (err.response.status === 409) {
          setError("Username or email already exists. Please try a different one.");
        } else {
          setError("An error occurred on the server. Please try again later.");
        }
      } else if (err.request) {
        // The request was made but no response was received
        setError("No response from server. Please check your internet connection.");
      } else {
        // Something happened in setting up the request that triggered an Error
        setError("An unexpected error occurred. Please try again.");
      }
    }
  };

  return (
    <div className="login-root">
      <div className="box-root flex-flex flex-direction--column" style={{ minHeight: '100vh', flexGrow: 1 }}>
        <div className="loginbackground box-background--white padding-top--64">
          <div className="loginbackground-gridContainer">
            <div className="box-root flex-flex" style={{ gridArea: 'top / start / 8 / end' }}>
              <div className="box-root" style={{ backgroundImage: 'linear-gradient(white 0%, rgb(247, 250, 252) 33%)', flexGrow: 1 }}></div>
            </div>
            <div className="box-root flex-flex" style={{ gridArea: '4 / 2 / auto / 5' }}>
              <div className="box-root box-divider--light-all-2 animationLeftRight tans3s" style={{ flexGrow: 1 }}></div>
            </div>
            <div className="box-root flex-flex" style={{ gridArea: '6 / start / auto / 2' }}>
              <div className="box-root box-background--blue800" style={{ flexGrow: 1 }}></div>
            </div>
            <div className="box-root flex-flex" style={{ gridArea: '7 / start / auto / 4' }}>
              <div className="box-root box-background--blue animationLeftRight" style={{ flexGrow: 1 }}></div>
            </div>
            <div className="box-root flex-flex" style={{ gridArea: '8 / 4 / auto / 6' }}>
              <div className="box-root box-background--gray100 animationLeftRight tans3s" style={{ flexGrow: 1 }}></div>
            </div>
            <div className="box-root flex-flex" style={{ gridArea: '2 / 15 / auto / end' }}>
              <div className="box-root box-background--cyan200 animationRightLeft tans4s" style={{ flexGrow: 1 }}></div>
            </div>
            <div className="box-root flex-flex" style={{ gridArea: '3 / 14 / auto / end' }}>
              <div className="box-root box-background--blue animationRightLeft" style={{ flexGrow: 1 }}></div>
            </div>
            <div className="box-root flex-flex" style={{ gridArea: '4 / 17 / auto / 20' }}>
              <div className="box-root box-background--gray100 animationRightLeft tans4s" style={{ flexGrow: 1 }}></div>
            </div>
            <div className="box-root flex-flex" style={{ gridArea: '5 / 14 / auto / 17' }}>
              <div className="box-root box-divider--light-all-2 animationRightLeft tans3s" style={{ flexGrow: 1 }}></div>
            </div>
          </div>
        </div>
        <div className="box-root padding-top--24 flex-flex flex-direction--column" style={{ flexGrow: 1, zIndex: 9 }}>
          <div className="box-root padding-top--48 padding-bottom--24 flex-flex flex-justifyContent--center">
            <h1><a href="#" rel="dofollow">Patient Classification</a></h1>
          </div>
          <div className="formbg-outer">
            <div className="formbg">
              <div className="formbg-inner padding-horizontal--48">
                <span className="padding-bottom--15">Sign in to your account</span>
                <form onSubmit={handleSubmit} className="form-container">
                  <div className="field padding-bottom--24">
                    <label htmlFor="username">Username</label>
                    <input
                      type="text"
                      name="username"
                      placeholder="Username"
                      value={formData.username}
                      onChange={handleChange}
                      required
                    />
                  </div>
                  {method === "register" && (
                    <div className="field padding-bottom--24">
                      <label htmlFor="email">Email</label>
                      <input
                        type="email"
                        name="email"
                        placeholder="Email"
                        value={formData.email}
                        onChange={handleChange}
                        required
                      />
                    </div>
                  )}
                  <div className="field padding-bottom--24">
                    <label htmlFor="password">Password</label>
                    <input
                      type="password"
                      name="password"
                      placeholder="Password"
                      value={formData.password}
                      onChange={handleChange}
                      required
                    />
                  </div>
                  <div className="field padding-bottom--24">
                    <input type="submit" name="submit" value={method === "register" ? "Register" : "Login"} />
                  </div>
                  {error && <p className="error">{error}</p>}
                </form>
              </div>
            </div>
            <div className="footer-link padding-top--24">
              <span>Don't have an account? <Link to="/Register">Sign up</Link></span>
              <div className="listing padding-top--24 padding-bottom--24 flex-flex center-center">
                <span><a href="#">Abdul Rahman</a></span>
                <span><a href="#">Ismath Zehra</a></span>
                <span><a href="#"> Mir Ali Abbas</a></span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Form;
