import requests

import streamlit as st
st.title("Stock Prediction")
name = st.text_input("Enter the name of the company: ")
stock = st.text_input("Enter the stock symbol: ")

if st.button("Predict"):

    st.write("Fetching news articles...")
    API_KEY = st.secrets["NEWS_API_KEY"]
    url = f"https://newsapi.org/v2/everything?q={name}&apiKey={API_KEY}"
    response = requests.get(url)
    articles = response.json()["articles"][:5] #get the first 5 articles
    st.write("Articles fetched successfully")

    st.write("Cleaning text...")
    import nltk #nltk is a library for natural language processing
    nltk.download('stopwords')
    from nltk.corpus import stopwords #stop words are common words that are not useful for sentiment analysis
    import re #re is a library for regular expressions

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english')) 

    def clean_text(text): #Not needed for transformer model - Doing it to show how to clean text
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = text.lower()
        text = ' '.join(word for word in text.split() if word not in stop_words)
        return text

    headlines = [clean_text(a["title"]) for a in articles] #clean the text of the articles
    st.write("Text cleaned successfully") 

    st.write("Running sentiment analysis...")
    from transformers import pipeline #pipeline is a function that allows you to use a pre-trained model for sentiment analysis
    classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert") #ProsusAI/finbert is a pre-trained model for sentiment analysis (returns a dictionary with 'label' and 'score')

    def get_sentiment_value(sentiment):
        if sentiment['label'] == "positive":
            return sentiment["score"]  # Positive sentiment: return score as is
        elif sentiment['label'] == "negative":  
            return -sentiment["score"]  # Negative sentiment: return negative score
        else:
            return 0 

    # Run the sentiment classifier on all headlines to get sentiment results for each headline
    sentiment_results = classifier(headlines) 

    # Convert each sentiment result to a numeric value using the function above
    sentiment_values = [get_sentiment_value(sentiment) for sentiment in sentiment_results] #use our function to convert the sentiment results to a numeric value

    # Calculate the average sentiment value for all headlines (assumed to be for the most recent day)
    avg = sum(sentiment_values) / len(sentiment_values)  # This gives a single sentiment score for the latest day
    print("AVG: ", avg)  
    st.write("Sentiment analysis completed successfully")

    for headline, sentiment in zip(headlines, sentiment_results): #zip is a function that combines two lists into a list of tuples (something new)
        
        print(f"Headline: {headline}")  #for debugging
        print(f"Sentiment: {sentiment}") 
        print("-")
        st.write(f"Headline: {headline}") #for frontend
        st.write(f"Sentiment: {sentiment}")
        st.write("-")
        

    st.write("Fetching stock data...")
    import yfinance as yf  #yahoo finance
    import pandas as pd  # Used for data manipulation and analysis

    # Define the stock ticker symbol you want to analyze
    ticker = stock  #Stock was taken by the user earlier 

    data = yf.download(ticker, period="5y", interval="1d")
    st.write("Stock data fetched successfully")
    # Flatten columns if they are multi-indexed. This ensures you can access columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    st.write("Printing stock data...")
    print(data.head())  # Print the first few rows of the downloaded data for debugging
    print("---------------------------")
    print(data.tail())  #Same debugging purpose 

    st.write(data.tail()) #for frontend - show the user the most recent data the code works with 


    #Plotting the stock price trend (general plotting code)
    st.write("Stock Price Trend:")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.index, data['Close'], label='Close Price', color='blue')
    ax.set_title(f'{stock} Stock Price Trend')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig) #Given to the user for inspection

    st.write("Adding sentiment to stock data...")
    sentiment_series = [0] * (len(data) - 1) + [avg]  # List of 0s with avg at the end (to make a coloumn in the data frame table used by the program)
    st.write("Sentiment added to stock data successfully")
    data['Sentiment'] = sentiment_series  # Add as a new column to the DataFrame
    data["Tomorrow"] = data["Close"].shift(-1)  # Shift the 'Close' column up by one to get the next day's closing price (very important - extremely inaccurate and buggy without it)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)  # 1 if tomorrow's close is higher than today's, else 0

    data.dropna(subset=["Tomorrow"], inplace=True)  # Remove rows where 'Tomorrow' is NaN (usually the last row)

    data['Return_1d'] = data['Close'].pct_change() # 1-day return
    # For Return_5d, MA_5, MA_10, MA_Ratio, Range_Pct, Vol_Pct, all calculations are based on current and past data only
    data['Return_5d'] = data['Close'].pct_change(5) # 5-day return
    data['MA_5'] = data['Close'].rolling(5).mean() # 5-day moving average
    data['MA_10'] = data['Close'].rolling(10).mean() # 10-day moving average
    data['MA_Ratio'] = data['MA_5'] / data['MA_10']  # Ratio of short-term to long-term MA
    data['Range_Pct'] = (data['High'] - data['Low']) / data['Close'] # Daily range as percent of close
    data['Vol_Pct'] = data['Volume'] / data['Volume'].rolling(5).mean() # Volume as percent of 5-day average

    data = data.dropna()  # Drop any rows with NaN values from feature engineering (important)

    features = ['Close', 'Return_1d', 'Return_5d', 'MA_5', 'MA_10', 'MA_Ratio', 'Range_Pct', 'Vol_Pct', 'Sentiment'] #features - used by the model to make a prediction
    X = data[features]  # Feature matrix (input variables for the model)
    Y = data["Target"]  # Target vector (output variable: 1 for up, 0 for down)

    from sklearn.preprocessing import StandardScaler # For standardizing the features
    scaler = StandardScaler() 
    X_scaled = scaler.fit_transform(X) # Standardize the features

    from sklearn.model_selection import train_test_split 
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, shuffle=False) # Split the data into training and testing sets

    from sklearn.linear_model import LogisticRegression 
    model = LogisticRegression(max_iter=500) # Create a logistic regression model - max_iter is the number of iterations to converge
    model.fit(X_train, Y_train) # Train the model using the training data

    from sklearn.metrics import confusion_matrix #For debugging purposes

    Y_pred = model.predict(X_test) # Make predictions on the test data

    print("\nConfusion Matrix:\n", confusion_matrix(Y_test, Y_pred)) # Print the confusion matrix - debugging purposes - to check how well the model is doing 

    accuracy = model.score(X_test, Y_test) # Evaluate the model's performance on the test data
    print(f"Accuracy: {accuracy}") # Print the accuracy of the model

    # Use the last row of X_scaled for prediction (latest available data)
    latest_price = X_scaled[-1].reshape(1,-1) # Get the last row of the scaled feature matrix and reshape it to a 2D array
    prediction = model.predict(latest_price) # Make a prediction using the model
    print(f"Prediction for tomorrow: ", "Up" if prediction[0] == 1 else "Down")

    st.write("Confusion Matrix:")
    st.write(confusion_matrix(Y_test, Y_pred))
    st.write(f"Accuracy {accuracy}")
    if prediction[0] == 1:
        p = "Up"
    else:
        p = "Down"

    # used to design some aspects of the streamlit frontend
    st.markdown(f"""
    <div style="
        border: 2px solid #880808;
        border-radius: 10px;
        padding: 15px;
        background-color: #f9f9f9;
        color: black;
        cursor: pointer;
        font-size: 18px;">
        Prediction for tomorrow: {p}
    </div>
""", unsafe_allow_html=True)

