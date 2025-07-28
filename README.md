Stock Movement Prediction App

This is a Streamlit-based web application that predicts the movement of a stock (up or down) based on recent news sentiment and historical stock closing prices. It uses the News API to fetch articles, applies Natural Language Processing (NLP) for sentiment analysis, incorporates stock price data, and outputs the predicted market direction.

Live App - https://stock-movement-prediction-fhfktrtnopkzigqssdxod9.streamlit.app/

-How it works

Input:

Company name (e.g., Tesla)

Stock symbol (e.g., TSLA)

-Data Collection:

Fetches the latest news articles related to the company using NewsAPI

Retrieves historical stock price data using Yahoo Finance for the given symbol

-Text Cleaning and Preprocessing:

Cleans headlines and descriptions by removing links, punctuation, stopwords, and extra spaces

Applies tokenization and normalization using the nltk library

-Sentiment Analysis:

Assigns polarity scores to each news item using rule-based sentiment analysis

Computes a weighted average sentiment score across all news articles

-Price-Based Prediction Logic:

Retrieves the most recent 5 days of closing stock prices

Calculates a moving average from the last 5 days

Compares this average with the current closing price to assess trend direction

-Final Prediction:

Combines the sentiment score and price trend information

Predicts whether the stock is likely to move up or down

-Tech Stack

Frontend: Streamlit
Backend: Python
APIs: NewsAPI, Yahoo Finance
NLP: nltk
Deployment: Streamlit Cloud

-Setup Instructions

Clone the repository:

git clone https://github.com/yourusername/stock-movement-prediction.git
cd stock-movement-prediction

-Install dependencies:

pip install -r requirements.txt

-Add your API key securely:

Create a file named .streamlit/secrets.toml and add the following line:
NEWS_API_KEY = "api key"

Run the app locally:

streamlit run main.py

To deploy:

Upload your repo to GitHub and connect it to Streamlit Cloud. Add your API key under the app’s "Secrets" tab on Streamlit Cloud for secure access.
