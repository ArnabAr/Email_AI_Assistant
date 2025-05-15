from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from logic import summarize_email_thread, generate_email_response, evaluate_summary_quality, evaluate_response_quality

app = FastAPI(title="Email Assistant")

class ThreadInput(BaseModel):
    emails: List[str]
    evaluate: Optional[bool] = False

class EmailInput(BaseModel):
    email: str
    evaluate: Optional[bool] = False

@app.post("/summarize/")
def summarize(input_data: ThreadInput):
    summary = summarize_email_thread(input_data.emails)
    result = {"summary": summary}
    if input_data.evaluate:
        evaluation = evaluate_summary_quality(input_data.emails, summary)
        result["evaluation"] = evaluation
    return result

@app.post("/respond/")
def respond(input_data: EmailInput):
    response = generate_email_response(input_data.email)
    result = {"response": response}
    if input_data.evaluate:
        evaluation = evaluate_response_quality(input_data.email, response)
        result["evaluation"] = evaluation
    return result

@app.get("/")
def root():
    return {"message": "Welcome to the Email Assistant API!"}
