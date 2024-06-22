import React from "react";
import Form from "../components/Form";

function Login() {
  return <Form route="/api/token/" method="POST" />;
}

export default Login;
