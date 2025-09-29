from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os, json

from api.predict import StyleInferenceService

QUIZ_PATH = "quiz_questions.json"
MODEL_DIR = "api/models_bundles"
SCHEMA_PATH = os.path.join(MODEL_DIR, "feature_columns.json")

with open(QUIZ_PATH, "r") as f:
    quiz_questions = json.load(f)

svc = StyleInferenceService(model_dir=MODEL_DIR, schema_path=SCHEMA_PATH)

app = FastAPI(title="EduBuddy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ActivityFeatures(BaseModel):
    weekly_click_slope: float
    wk_entropy_slope: float
    content_prop: float
    forum_prop: float
    quiz_prop: float
    url_prop: float
    nav_entropy: float
    total_clicks: int

class SubmitRequest(BaseModel):
    student_id: str
    name: str
    quiz_answers: list[int]
    activity_features: ActivityFeatures

@app.get("/api/quiz-questions")
def get_quiz_questions():
    return quiz_questions

@app.post("/api/submit-quiz")
def submit_quiz(req: SubmitRequest):
    style_labels = ["Visual", "Auditory", "Kinesthetic"]
    style_counts = {k: 0 for k in style_labels}
    for idx, ans in enumerate(req.quiz_answers):
        style = quiz_questions[idx]["options"][ans]["style"]
        style_counts[style] += 1
    total = sum(style_counts.values())
    quiz_percentages = {k: int(v * 100 / total) for k, v in style_counts.items()}

    df = pd.DataFrame([req.activity_features.dict()])
    ai_pred = svc.predict(df).iloc[0]
    tags = ["visual_verbal", "active_reflective", "global_sequential"]
    style_map = {"visual_verbal": "Visual", "active_reflective": "Auditory", "global_sequential": "Kinesthetic"}
    ai_percentages = {style_map[tag]: int(ai_pred[f"proba_{tag}"] * 100) for tag in tags}
    ai_style = max(ai_percentages, key=ai_percentages.get)

    quiz_style = max(quiz_percentages, key=quiz_percentages.get)
    comparison = "Match" if quiz_style == ai_style else "Mismatch"

    strategies = {
        "Visual": ["Create a concept map", "Read & write reflections", "Step-by-step checklist"],
        "Auditory": ["Discuss with peers", "Listen to recorded lectures", "Use mnemonics"],
        "Kinesthetic": ["Hands-on practice", "Use interactive tools", "Build models"],
    }

    activity_log = [
        "Quiz completed",
        "Forum post",
        "Wiki edited",
        "Last login: 2025-09-27"
    ]

    return {
        "student_id": req.student_id,
        "name": req.name,
        "quiz_result": quiz_percentages,
        "ai_result": ai_percentages,
        "ai_style": ai_style,
        "quiz_style": quiz_style,
        "comparison": comparison,
        "recommended": strategies[ai_style],
        "activity_log": activity_log
    }