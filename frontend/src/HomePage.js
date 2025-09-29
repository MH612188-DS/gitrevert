import React, { useState } from "react";

export default function HomePage({ onStart }) {
  const [student_id, setStudentId] = useState("");
  const [name, setName] = useState("");
  return (
    <div style={{ textAlign: "center", marginTop: "60px" }}>
      <h1>EduBuddy</h1>
      <input
        placeholder="Student ID"
        value={student_id}
        onChange={e => setStudentId(e.target.value)}
        style={{ margin: "10px" }}
      />
      <input
        placeholder="Name"
        value={name}
        onChange={e => setName(e.target.value)}
        style={{ margin: "10px" }}
      />
      <br />
      <button
        onClick={() => onStart({ student_id, name })}
        style={{ fontSize: "20px", marginTop: "10px" }}
        disabled={!student_id || !name}
      >
        Start Test
      </button>
    </div>
  );
}