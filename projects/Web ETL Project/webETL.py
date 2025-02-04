# Filename: webETL.py
# Author: Nimalan Subramanian
# Created: 2025-01-29
# Description: Extract data from web source, transform to structured format then load into database/data lake.

import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from collections import Counter
import pandas as pd
import nltk
import ssl

#Bypass SSL Verification
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')

#Extract text from web article
class WebScraper:
    def __init__(self, url):
        self.url = url

    def extract_article_text(self):
        response = requests.get(self.url)
        html_content = response.content
        soup = BeautifulSoup(html_content, "html.parser")
        article_text = soup.get_text()
        return article_text
    
#Clean and preprocess extracted text
class TextProcessor:
    def __init__(self, nltk_stopwords):
        self.nltk_stopwords = nltk_stopwords

    def tokenize_and_clean(self, text):
        words = text.split()
        filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in self.nltk_stopwords]
        return filtered_words
    
#Define ETL pipeline class
class ETLPipeline:
    def __init__(self, url):
        self.url = url
        self.nltk_stopwords = set(stopwords.words("english"))

    def run(self):
        scraper = WebScraper(self.url)
        article_text = scraper.extract_article_text()

        processor = TextProcessor(self.nltk_stopwords)
        filtered_words = processor.tokenize_and_clean(article_text)

        word_freq = Counter(filtered_words)
        df = pd.DataFrame(word_freq.items(), columns=["Words", "Frequencies"])
        df = df.sort_values(by="Frequencies", ascending=False)
        return df
    
#Scrape text data from an article from web and count word frequency
if __name__ == "__main__":
    article_url = "https://www.informatica.com/resources/articles/what-is-etl-pipeline.html"
    pipeline = ETLPipeline(article_url)
    result_df = pipeline.run()
    print(result_df.head())