<<<<<<< HEAD
# Stock Movement Prediction App

A Streamlit-based web application that predicts stock movement (up or down) based on recent news sentiment analysis and historical stock price data. The app combines natural language processing with financial data to provide market direction predictions.

**Live App:** [https://stock-movement-prediction-fhfktrtnopkzigqssdxod9.streamlit.app/](https://stock-movement-prediction-fhfktrtnopkzigqssdxod9.streamlit.app/)

## Features

- **Real-time News Analysis**: Fetches latest news articles related to specified companies
- **Sentiment Analysis**: Uses FinBERT model for financial sentiment analysis
- **Stock Data Integration**: Retrieves historical price data from Yahoo Finance
- **Machine Learning Prediction**: Combines sentiment and technical indicators for predictions
- **Interactive Web Interface**: User-friendly Streamlit frontend
- **Live Deployment**: Accessible via Streamlit Cloud

## How It Works

### Input
- **Company name** (e.g., Tesla)
- **Stock symbol** (e.g., TSLA)

### Data Collection Process

1. **News Articles**: Fetches the latest 5 news articles related to the company using NewsAPI
2. **Stock Data**: Retrieves 5 years of historical stock price data using Yahoo Finance

### Text Processing Pipeline

1. **Text Cleaning**: 
   - Removes punctuation, special characters, and links
   - Converts text to lowercase
   - Removes stopwords using NLTK

2. **Sentiment Analysis**:
   - Uses FinBERT (ProsusAI/finbert) model for financial sentiment analysis
   - Assigns polarity scores to each news headline
   - Computes weighted average sentiment across all articles

### Technical Analysis

The app calculates various technical indicators:
- **Moving Averages**: 5-day and 10-day moving averages
- **Price Returns**: 1-day and 5-day returns
- **Volume Analysis**: Volume as percentage of 5-day average
- **Price Range**: Daily range as percentage of close price
- **MA Ratio**: Short-term to long-term moving average ratio

### Prediction Logic

1. **Feature Engineering**: Combines sentiment scores with technical indicators
2. **Model Training**: Uses Logistic Regression on historical data
3. **Prediction**: Outputs binary prediction (Up/Down) for next trading day
4. **Performance Metrics**: Displays confusion matrix and accuracy scores

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **APIs**: 
  - NewsAPI (news sentiment)
  - Yahoo Finance (stock data)
- **NLP**: 
  - NLTK (text preprocessing)
  - Transformers (FinBERT sentiment analysis)
- **Machine Learning**: 
  - Scikit-learn (Logistic Regression)
  - Pandas (data manipulation)
- **Deployment**: Streamlit Cloud

## Setup Instructions

### Prerequisites
- Python 3.7+
- pip package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/stock-movement-prediction.git
   cd stock-movement-prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API key**:
   - Create a file named `.streamlit/secrets.toml`
   - Add your NewsAPI key:
     ```toml
     NEWS_API_KEY = "your_api_key_here"
     ```

4. **Run the app locally**:
   ```bash
   streamlit run main.py
   ```

### Getting API Keys

1. **NewsAPI**: 
   - Visit [newsapi.org](https://newsapi.org)
   - Sign up for a free account
   - Copy your API key

## Deployment

### Streamlit Cloud Deployment

1. **Upload to GitHub**:
   - Push your code to a GitHub repository
   - Ensure `requirements.txt` is included

2. **Connect to Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository

3. **Configure Secrets**:
   - In your Streamlit Cloud app settings
   - Go to "Secrets" tab
   - Add your `NEWS_API_KEY` securely

4. **Deploy**:
   - Click "Deploy" to make your app live
   - Your app will be accessible via the provided URL
=======
