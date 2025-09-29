import React, { useEffect } from "react";
import { useState } from "react";

function ChartComponent({ data }) {
  if (!data) return null;
  return (
    <div style={{ margin: "30px 0" }}>
      <h3>Learning Style Chart</h3>
      <div style={{ display: "flex", justifyContent: "space-around" }}>
        {Object.entries(data).map(([style, pct]) => (
          <div key={style} style={{ textAlign: "center" }}>
            <div style={{
              height: `${pct * 2}px`,
              width: "40px",
              background: "#4b7bec",
              marginBottom: "5px",
              transition: "height 0.6s"
            }} />
            <div>{style}<br />{pct}%</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ProfilePage({
  studentInfo,
  quizAnswers,
  activityFeatures,
  profile,
  setProfile,
  onRestart,
}) {
  const [loading, setLoading] = useState(!profile);

  useEffect(() => {
    if (!profile) {
      setLoading(true);
      fetch("/api/submit-quiz", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          student_id: studentInfo.student_id,
          name: studentInfo.name,
          quiz_answers: quizAnswers,
          activity_features: activityFeatures,
        }),
      })
        .then(res => res.json())
        .then((data) => {
          setProfile(data);
          setLoading(false);
        });
    }
  }, [profile, studentInfo, quizAnswers, activityFeatures, setProfile]);

  if (loading || !profile) return <div>Loading...</div>;

  return (
    <div style={{ maxWidth: "700px", margin: "auto" }}>
      <h2>Student Profile</h2>
      <div><b>Student ID:</b> {profile.student_id}</div>
      <div><b>Name:</b> {profile.name}</div>
      <h3>Quiz Result:</h3>
      <ul>
        {Object.entries(profile.quiz_result).map(([style, pct]) => (
          <li key={style}>{style}: {pct}%</li>
        ))}
      </ul>
      <h3>AI Predicted Style:</h3>
      <div>{profile.ai_style} ({profile.ai_result[profile.ai_style]}%)</div>
      <h3>Comparison:</h3>
      <div>{profile.comparison}</div>
      <h3>Recommended Strategies:</h3>
      <ul>
        {profile.recommended.map((s, idx) => <li key={idx}>{s}</li>)}
      </ul>
      <ChartComponent data={profile.quiz_result} />
      <h3>Activity Log</h3>
      <ul>
        {profile.activity_log.map((act, idx) => <li key={idx}>{act}</li>)}
      </ul>
      <button onClick={onRestart} style={{ marginTop: "25px" }}>
        Restart
      </button>
    </div>
  );
}