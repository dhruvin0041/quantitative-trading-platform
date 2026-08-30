# src/features/nlp_processor.py
from transformers import BertTokenizer
import requests
import xml.etree.ElementTree as ET
import os
from google import genai


import time
import json
import re


class GeminiAnalyzer:
    def __init__(self, api_key=None):
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
            self.model_name = "gemini-3-flash-preview"
            self.active = True
        else:
            self.active = False

    def analyze_fundamental_alpha(self, news_context: str, ticker: str):
        """
        Uses Gemini to perform qualitative alpha extraction with retry logic for free tier.
        """
        if not self.active:
            return 0.0, "Qualitative analysis unavailable"

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

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                text = response.text

                match = re.search(r"\{.*\}", text, re.DOTALL)
                if match:
                    data = json.loads(match.group())
                    return float(data.get("score", 0.0)), data.get(
                        "reasoning", "Analysis complete."
                    )
                return 0.0, "Qualitative analysis unavailable"
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg and attempt < max_retries:
                    time.sleep(5)  # Wait 5 seconds as requested for rate limits
                    continue
                return 0.0, "Qualitative analysis unavailable"

        return 0.0, "Qualitative analysis unavailable"


class NewsTokenizer:
    def __init__(self, max_length=128):
        # FinBERT uses the standard BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
        self.max_length = max_length

    def fetch_sec_filings(self, ticker: str):
        """
        Fetches the latest 8-K or 10-Q filings from the SEC EDGAR RSS feed.
        """
        try:
            # SEC requires a descriptive User-Agent
            headers = {"User-Agent": "HydraTerminal/1.0 (contact: research@hydra.ai)"}
            url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=8-K&output=atom"
            response = requests.get(url, headers=headers, timeout=5)

            if response.status_code == 200:
                root = ET.fromstring(response.content)
                # Extract first entry's title as a summary of the filing
                for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
                    title = entry.find("{http://www.w3.org/2005/Atom}title").text
                    return f"SEC FILING ALERT: {title}"
            return ""
        except Exception:
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

        encoded = self.tokenizer(
            combined_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="np",
        )
        return encoded["input_ids"][0], encoded["attention_mask"][0], combined_text
