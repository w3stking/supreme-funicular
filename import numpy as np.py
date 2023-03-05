import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import matplotlib.pyplot as plt

# Define a function to get the stock data and train the model
def analyze_stock(symbol):
    # Set the date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    # Load the data
    data = yf.download(symbol, start=start_date, end=end_date)

    # Check if data is available for at least one month
    if len(data) < 20:
        print("Error: Not enough data available for analysis")
        return

    # Perform data cleaning, feature engineering, and preprocessing as necessary
    # Add additional features such as volume, bid-ask spread, and short interest
    data['Volume'] = data['Volume'].fillna(data['Volume'].mean())
    data['Bid-Ask Spread'] = data['High'] - data['Low']
    data['Short Interest'] = data['High'] - data['Low']
    import ta
    #data['RSI'] = ta.rsi(data['Close'], timeperiod=14)
    #data['MACD'] = ta.macd(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)[0]
    #data['ATR'] = ta.average_true_range(data['High'], data['Low'], data['Close'], timeperiod=14)

    # Create a binary target variable indicating whether the price will go up or down
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

    # Split data into training and validation sets
    train_data, val_data = train_test_split(data, test_size=0.2, shuffle=False)

    # Scale the data
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data.drop(columns=['Target']))
    val_data_scaled = scaler.transform(val_data.drop(columns=['Target']))

    # Reshape data for the bidirectional LSTM model
    train_data_scaled = train_data_scaled.reshape(train_data_scaled.shape[0], train_data_scaled.shape[1], 1)
    val_data_scaled = val_data_scaled.reshape(val_data_scaled.shape[0], val_data_scaled.shape[1], 1)

    # Define the bidirectional LSTM model
    model = Sequential()
    model.add(tf.keras.layers.Bidirectional(LSTM(units=50, return_sequences=True), input_shape=(train_data_scaled.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(tf.keras.layers.Bidirectional(LSTM(units=50, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(tf.keras.layers.Bidirectional(LSTM(units=50, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(tf.keras.layers.Bidirectional(LSTM(units=50)))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))
    # Add traditional statistical methods
    model.add(Dense(units=1, activation='linear'))
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy')

    # Train the model
    history = model.fit(train_data_scaled, train_data['Target'], epochs=500, batch_size=32, validation_data=(val_data_scaled, val_data['Target']))

    # Make predictions
    predictions = model.predict(val_data_scaled)
    predictions = np.where(predictions > 0.5, 1, 0)
    target_price = val_data['Close'].shift(-1).values
    
    # Print bearish/bullish message based on prediction
    if predictions[0] == 1:
        print("The stock is predicted to be bullish for the next day")
    else:
        print("The stock is predicted to be bearish for the next day")

    # Print target price
    print(f"Target price for the next day: {target_price[0]:.2f}")

    # Evaluate the model
    accuracy = accuracy_score(val_data['Target'], predictions)
    print(f"Accuracy: {accuracy}")

    # Visualize the results
    plt.figure(figsize=(15,7))
    plt.plot(val_data.index, val_data['Close'], label='Close Price')
    plt.plot(val_data.index, predictions, label='Predicted Direction', linestyle='--')
    plt.legend()
    plt.xticks(rotation=0)
    plt.show()
    
    # Get the last 5 working days that stock market is open
    last_five_days = data.tail(5)
    print('Last 5 Working Days that Stock Market is open:')
    print(last_five_days[['Open', 'High', 'Low', 'Close']])
    
    # Plot the last 5 days of stock market data
    plt.figure(figsize=(15,7))
    plt.plot(last_five_days.index, last_five_days['Open'], label='Opening Price')
    plt.plot(last_five_days.index, last_five_days['High'], label='High Price')
    plt.plot(last_five_days.index, last_five_days['Low'], label='Low Price')
    plt.plot(last_five_days.index, last_five_days['Close'], label='Closing Price')
    plt.legend()
    plt.xticks(rotation=0)
    plt.show()

# Get the stock symbol from the user
symbol = input("Enter a stock symbol to analyze: ")

# Update the stock data if it was requested in the past
stock = yf.Ticker(symbol)
last_data_date = stock.history(period="1d").index[0]
if last_data_date.tzinfo is None:
    last_data_date = last_data_date.tz_localize('UTC')
if last_data_date < pd.Timestamp.today().tz_localize('UTC').normalize():
    print("Updating stock data...")
    stock.history(period="max")

# Analyze the stock
import tensorflow as tf
analyze_stock(symbol)