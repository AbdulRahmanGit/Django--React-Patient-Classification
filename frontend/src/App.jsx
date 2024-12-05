// src/App.jsx
import React from "react";

import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Login from "./pages/Login";
import Home from "./pages/Home";
import Register from "./pages/Register";
import MLResults from "./pages/MLResults"; // Import MLResults component
import ProtectedRoute from "./components/ProtectedRoute";
import Predict from "./pages/Predict";
import Dataset from "./pages/Dataset";
import { inject } from '@vercel/analytics';
 
inject();

function Logout() {
  localStorage.clear();
  return <Navigate to="/login" />;
}

function RegisterAndLogout() {
  localStorage.clear();
  return <Register />;
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <Home />
            </ProtectedRoute>
          }
        />
        <Route
          path="/MLResults"
          element={
            <ProtectedRoute>
              <MLResults />
            </ProtectedRoute>
          }
        />
        <Route
          path="/Dataset"
          element={
            <ProtectedRoute>
              <Dataset />
            </ProtectedRoute>
          }
        />
        <Route
          path="/Predict"
          element={
            <ProtectedRoute>
              <Predict />
            </ProtectedRoute>
          }
        />
        <Route path="/login" element={<Login />} />
        <Route path="/logout" element={<Logout />} />
        <Route path="/register" element={<RegisterAndLogout />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
