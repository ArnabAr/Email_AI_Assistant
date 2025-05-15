import os
import re
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime
from typing import List, Dict, Tuple
import ast

# Constants
DATASET_PATH = "emails.csv"  # Update with your dataset path
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load .env
load_dotenv()

# Get the API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
gemini_model = genai.GenerativeModel('models/gemini-1.5-flash-latest')
gemini_evaluation_model = genai.GenerativeModel('models/gemini-2.5-flash-preview-04-17')

def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded with {len(df)} emails")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def preprocess_email(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"(?i)^((Message-ID|Date|From|To|Subject|Cc|Bcc|Mime-Version|Content-Type|Content-Transfer-Encoding|X-[\w-]+):\s.*\n)+", '', text)
    text = re.sub(r"(?i)(\n)?(--|__|Regards,|Best,|Thanks,|Sincerely).*", '', text)
    text = re.sub(r"(?s)(>+\s?.*\n)+", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    return text

def summarize_email_thread(emails):
    combined_text = "\n\n".join([preprocess_email(email) for email in emails])
    if len(combined_text.split()) < 50:
        return "Thread too short for meaningful summary."

    prompt = f"""
    You are a professional assistant tasked with summarizing a multi-message email thread.

    Your goal is to provide a clear, actionable summary suitable for busy professionals. Focus strictly on the following:
    - **Key decisions made** (if any)
    - **Action items and owners** (who needs to do what)
    - **Core discussion points** (main topics or concerns raised)

    Guidelines:
    - Write in bullet points or a short structured paragraph.
    - Exclude greetings, signatures, and irrelevant small talk.
    - Do not include any meta-comments about summarizing.
    - Only return the summary content — no explanations or preambles.

    Email Thread:
    {combined_text}
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Summary generation failed."

def generate_email_response(email_text):
    cleaned_text = preprocess_email(email_text)
    email_type = identify_common_email_type(cleaned_text)

    prompt = f"""
    You are an executive assistant generating email replies in a professional business context.

    Objective:
    Write a polite, concise, and contextually appropriate 3-5 sentence response to the following email.

    Email type: **{email_type}**
    Tone: Professional and courteous  
    Length: 3-5 sentences  
    Constraints:
    - Do not repeat the sender's message.
    - Do not add generic disclaimers or unrelated content.
    - Focus on addressing the email's core intent clearly.

    Email Content:
    """
    {cleaned_text[:5000]}
    """

    Your Response:
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Response generation failed."

def identify_common_email_type(email_text):
    email_text = email_text.lower()
    keyword_map = {
        "meeting request": ['meeting', 'schedule', 'appointment', 'calendar', 'reschedule', 'invite'],
        "status update": ['status', 'update', 'progress', 'report', 'milestone', 'summary'],
        "information request": ['question', 'help', 'advice', 'suggestion', 'clarify', 'inquire'],
        "thank you note": ['thank', 'thanks', 'appreciate', 'grateful', 'gratitude'],
        "follow-up": ['follow up', 'just checking', 'circling back', 'ping', 'reminder'],
        "issue report": ['bug', 'issue', 'error', 'problem', 'fail', 'failure', 'glitch'],
        "approval request": ['approve', 'approval', 'authorize', 'permission', 'sign off']
    }
    for category, keywords in keyword_map.items():
        if any(kw in email_text for kw in keywords):
            return category
    return "general inquiry"

def evaluate_summary_quality(original_emails: List[str], summary: str) -> Dict[str, float]:
    combined_original = "\n\n".join([preprocess_email(email) for email in original_emails])
    prompt = f"""
    You are an expert communication analyst. Your task is to evaluate a **generated email summary** based on the content of the **original email threads**.

    Use the following criteria, scoring each from **0.0 to 1.0** (where 1.0 = perfect, 0.0 = poor):

    [SCORING CRITERIA]
    1. **main_points_coverage**
    2. **conciseness**
    3. **accuracy**
    4. **action_clarity**
    5. **structure**

    [INPUT DATA]
    Original Emails:
    {combined_original[:10000]}

    Generated Summary:
    {summary}

    Evaluation (return only a Python dictionary with float values from 0 to 1 and no other text):
    """
    try:
        response = gemini_evaluation_model.generate_content(prompt)
        print("\n--- Summary Evaluation:---\n", response.text)
        return ast.literal_eval(response.text.strip())
    except Exception as e:
        print(f"Error evaluating summary: {e}")
        return {key: 0.1 for key in ['main_points_coverage', 'conciseness', 'accuracy', 'action_clarity', 'structure']}

def evaluate_response_quality(original_email: str, response: str) -> Dict[str, float]:
    prompt = f"""
    You are an expert email communication evaluator.

    Evaluate the **generated email response** using the following five criteria, each scored 0.0 to 1.0:
    [RETURN FORMAT REQUIREMENT]
    Return ONLY a **valid Python dictionary** in this format:
    {{
        'coherence': float,
        'contextual_appropriateness': float,
        'relevance': float,
        'professionalism': float,
        'actionability': float
    }}

    [INPUT]
    Original Email:
    '{original_email[:5000]}'

    Generated Response:
    '{response}'

    [IMPORTANT]
    - DO NOT explain your ratings.
    - DO NOT return anything except the dictionary.
    """
    try:
        response = gemini_evaluation_model.generate_content(prompt)
        print("\n--- Response Evaluation:--- \n", response.text)
        return ast.literal_eval(response.text.strip())
    except Exception as e:
        print(f"Error evaluating response: {e}")
        return {key: 0.1 for key in ['coherence', 'contextual_appropriateness', 'relevance', 'professionalism', 'actionability']}

def format_evaluation_results(evaluation: Dict[str, float], prefix: str = "") -> str:
    formatted = []
    for key, value in evaluation.items():
        formatted_key = prefix + key.replace('_', ' ').title()
        formatted.append(f"{formatted_key}: {value:.2f}/1.00")
    return "\n".join(formatted)
