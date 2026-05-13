# src/features/nlp_processor.py
from transformers import BertTokenizer
import numpy as np
import requests
import xml.etree.ElementTree as ET

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