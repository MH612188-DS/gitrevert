import React, { useState } from "react";

const defaultFeatures = {
  weekly_click_slope: 0.0,
  wk_entropy_slope: 0.0,
  content_prop: 0.2,
  forum_prop: 0.1,
  quiz_prop: 0.3,
  url_prop: 0.1,
  nav_entropy: 0.5,
  total_clicks: 120,
};

export default function ActivityForm({ onSubmit }) {
  const [features, setFeatures] = useState(defaultFeatures);

  function handleChange(e) {
    const { name, value } = e.target;
    setFeatures(f => ({
      ...f,
      [name]: name === "total_clicks" ? parseInt(value) : parseFloat(value),
    }));
  }

  return (
    <div style={{ maxWidth: "500px", margin: "auto" }}>
      <h2>Enter Activity Features</h2>
      {Object.keys(defaultFeatures).map(key => (
        <div key={key} style={{ margin: "10px 0" }}>
          <label>{key}: </label>
          <input
            type="number"
            name={key}
            value={features[key]}
            onChange={handleChange}
            step="any"
          />
        </div>
      ))}
      <button
        onClick={() => onSubmit(features)}
        style={{ fontSize: "18px", marginTop: "15px" }}
      >
        Submit and View Profile
      </button>
    </div>
  );
}