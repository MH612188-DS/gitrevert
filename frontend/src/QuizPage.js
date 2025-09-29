import React, { useState, useEffect } from "react";

export default function QuizPage({ onSubmit }) {
  const [questions, setQuestions] = useState([]);
  const [answers, setAnswers] = useState([]);
  useEffect(() => {
    fetch("/api/quiz-questions")
      .then(res => res.json())
      .then(setQuestions);
    setAnswers(Array(10).fill(null));
  }, []);

  function handleAnswer(qIdx, aIdx) {
    const newAns = [...answers];
    newAns[qIdx] = aIdx;
    setAnswers(newAns);
  }

  return (
    <div style={{ maxWidth: "700px", margin: "auto" }}>
      <h2>Learning Style Test</h2>
      {questions.map((q, idx) => (
        <div key={idx} style={{ marginBottom: "20px" }}>
          <div>{idx + 1}. {q.question}</div>
          {q.options.map((opt, oidx) => (
            <label key={oidx} style={{ marginLeft: "15px", display: "block" }}>
              <input
                type="radio"
                checked={answers[idx] === oidx}
                onChange={() => handleAnswer(idx, oidx)}
              />
              {opt.text}
            </label>
          ))}
        </div>
      ))}
      <button
        onClick={() => onSubmit(answers)}
        disabled={answers.some(a => a === null)}
        style={{ fontSize: "18px" }}
      >
        Next: Enter Activity Features
      </button>
    </div>
  );
}