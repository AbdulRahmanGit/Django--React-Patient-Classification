import React from "react";
import Predict from "../components/Predict";

function PredictPage() {
  return <Predict route="/api/predictions/" method="POST" />;
}

export default PredictPage;
