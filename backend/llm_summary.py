# llm_summary.py
from transformers import pipeline
import os

# Use a small HF model; if you prefer no dependency set GENERATE=False.
try:
    summarizer = pipeline("text2text-generation", model="google/flan-t5-small")
except Exception as e:
    summarizer = None
    print("LLM summarizer unavailable:", e)

def generate_summary(modality, result):
    # result is a dict from inference_{breast,prostate}.py
    text = f"""This is a {modality} screening result.
Prediction: {result.get('prediction') or result.get('prediction','')}
Probability: {result.get('probability', result.get('positive_pct',0))}.
Please produce a short clinical-style summary (1-2 sentences), mention actionable next steps."""
    if summarizer:
        out = summarizer(text, max_length=64, do_sample=False)
        return out[0]["generated_text"]
    else:
        # fallback simple summary
        if modality.lower().startswith("breast"):
            prob = result.get("probability",0)
            pred = result.get("prediction","")
            return f"AI predicts {pred} with probability {prob:.2f}. Recommend radiologist review and possible biopsy if clinically indicated."
        else:
            pct = result.get("positive_pct",0)
            return f"AI identified suspicious region occupying {pct:.1f}% of prostate volume. Recommend radiologist correlation and targeted biopsy if clinically appropriate."



