# src/features/nlp_processor.py
from transformers import BertTokenizer
import numpy as np
import requests
import xml.etree.ElementTree as ET
import os
import google.generativeai as genai

class GeminiAnalyzer:
    def __init__(self, api_key=None):
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.active = True
        else:
            self.active = False

    def analyze_fundamental_alpha(self, news_context: str, ticker: str):
        """
        Uses Gemini to perform qualitative alpha extraction.
        Specifically looks for 'moving targets' (performance metric shifts).
        """
        if not self.active:
            return 0.0, "Gemini API Key missing. Skipping qualitative analysis."

        prompt = f"""
        Act as an institutional quantitative analyst. Analyze the following news and SEC filing context for {ticker}:
        
        '{news_context}'
        
        Task:
        1. Identify any 'moving targets' (Is the company shifting focus from one metric to another?).
        2. Identify hidden litigious risks or uncertainty language.
        3. Rate the 'Qualitative Alpha' from -1.0 (Strong Negative Narrative) to 1.0 (Strong Positive Narrative).
        
        Return ONLY a JSON object with:
        {{"score": float, "reasoning": "string (max 15 words)"}}
        """
        try:
            response = self.model.generate_content(prompt)
            # Simple extraction from JSON string
            text = response.text
            import json
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return float(data.get("score", 0.0)), data.get("reasoning", "Analysis complete.")
            return 0.0, "Could not parse LLM response."
        except Exception as e:
            return 0.0, f"Gemini Error: {str(e)}"

class NewsTokenizer:
    def __init__(self, max_length=128):
        # FinBERT uses the standard BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
        self.max_length = max_length

    def fetch_sec_filings(self, ticker: str):
        """
        Fetches the latest 8-K or 10-Q filings from the SEC EDGAR RSS feed.
        """
        try:
            # SEC requires a descriptive User-Agent
            headers = {'User-Agent': 'HydraTerminal/1.0 (contact: research@hydra.ai)'}
            url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=8-K&output=atom"
            response = requests.get(url, headers=headers, timeout=5)

            if response.status_code == 200:
                root = ET.fromstring(response.content)
                # Extract first entry's title as a summary of the filing
                for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                    title = entry.find('{http://www.w3.org/2005/Atom}title').text
                    return f"SEC FILING ALERT: {title}"
            return ""
        except:
            return ""

    def tokenize_daily_news(self, news_string: str, ticker: str = None):
        """
        Converts news + SEC filings into BERT input IDs and Attention Masks.
        """
        sec_text = ""
        if ticker:
            sec_text = self.fetch_sec_filings(ticker)

        combined_text = f"{news_string} {sec_text}".strip()

        # Handle cases where there is no news for the day
        if not combined_text:
            combined_text = "No significant financial news or SEC filings today."

        encoded = self.tokenizer.encode_plus(
            combined_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )
        return encoded['input_ids'][0], encoded['attention_mask'][0], combined_text