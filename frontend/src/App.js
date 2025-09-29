import React, { useState } from "react";
import HomePage from "./HomePage";
import QuizPage from "./QuizPage";
import ActivityForm from "./ActivityForm";
import ProfilePage from "./ProfilePage";

function App() {
  const [studentInfo, setStudentInfo] = useState(null);
  const [quizAnswers, setQuizAnswers] = useState(null);
  const [activityFeatures, setActivityFeatures] = useState(null);
  const [profile, setProfile] = useState(null);
  const [page, setPage] = useState("home");

  return (
    <div>
      {page === "home" && (
        <HomePage
          onStart={(info) => {
            setStudentInfo(info);
            setPage("quiz");
          }}
        />
      )}
      {page === "quiz" && (
        <QuizPage
          onSubmit={(answers) => {
            setQuizAnswers(answers);
            setPage("activity");
          }}
        />
      )}
      {page === "activity" && (
        <ActivityForm
          onSubmit={(features) => {
            setActivityFeatures(features);
            setPage("profile");
          }}
        />
      )}
      {page === "profile" && (
        <ProfilePage
          studentInfo={studentInfo}
          quizAnswers={quizAnswers}
          activityFeatures={activityFeatures}
          profile={profile}
          setProfile={setProfile}
          onRestart={() => {
            setPage("home");
            setStudentInfo(null);
            setQuizAnswers(null);
            setActivityFeatures(null);
            setProfile(null);
          }}
        />
      )}
    </div>
  );
}
export default App;