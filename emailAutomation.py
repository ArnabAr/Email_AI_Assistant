import os
import re
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Load dataset

def load_emails(filepath):
    df = pd.read_csv(filepath)
    df = df[['file', 'message']].dropna()
    df['message'] = df['message'].apply(clean_email_body)
    print(df['message'][1])
    return df

# 2. Clean email content
def clean_email_body(text):
    text = re.sub(r"(?i)(Message-ID|Date|From|To|Subject|Cc|Bcc|Mime-Version|Content-Type|Content-Transfer-Encoding):.*", '', text)
    text = re.sub(r"(?s)(>.*\n)+", '', text)  # remove quoted replies
    text = re.sub(r"(?i)(\n)*--.*", '', text)  # strip signature
    text = re.sub(r"\s+", ' ', text)
    return text.strip()
    


def main(filepath):
    # Load and prepare dataset
    emails = load_emails(filepath)

if __name__ == "__main__":
    main("emails.csv")