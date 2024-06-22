import React from "react";
import CsvViewer from "../components/CsvViewer";

function Dataset() {
  return <CsvViewer route="/api/view-dataset/" method="view-dataset" />;
}

export default Dataset;
